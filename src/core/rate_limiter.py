"""
Thread-safe weighted rate limiter for Binance public API.

Binance enforces a per-IP budget of 6000 request-weight per minute.
Render free-tier egress IPs are SHARED between services, so our bot must:
  1. Budget itself well below the ceiling (default 4500 weight/min)
  2. Track the server-reported `X-MBX-USED-WEIGHT-1M` header and back off
     when the SHARED IP is near the ceiling (neighbors' traffic counts too)
  3. Honor 429 `Retry-After` with a global cooldown instead of retry-storming

Used by BinanceClient before every outbound request.
"""
import json
import os
import threading
import time
from collections import deque
from pathlib import Path

from src.utils.logger import log

# v5.15: the cooldown survives process restarts. Binance 418 IP bans live on
# BINANCE'S side (minutes..days for repeat offenders) - a Render deploy or
# crash restart used to wipe our in-memory cooldown and the fresh process
# immediately hammered the still-banned IP (production 2026-09-26: restart
# during a ban -> /ticker/24hr -> fresh 418 -> 2476s re-cooldown). The state
# file stores an EPOCH deadline (monotonic clocks are per-process).
STATE_FILE = Path("data/rate_state.json")
# Hard sanity cap: Binance never documents bans longer than 3 days; honouring
# the server Retry-After fully (up to 24h) beats poking and EXTENDING the ban.
COOLDOWN_HARD_CAP_S = 86400.0

# v5.29 post-ban probe gate. Production 2026-10-02: a 1892s (escalated) 418
# ban on the shared Render IP. When a LONG server ban expires we used to
# release ALL callers at once (WS seeder + scalp tick + analysis cycle) into
# an IP that Binance may have silently EXTENDED the ban on - and every
# request sent during a ban extends it (the documented 598s -> 1892s
# escalation loop). Now a long ban arms a PROBE requirement: after expiry
# the first caller sends one weight-1 /time request; only a verified-clean
# IP releases the herd. Priority traffic (position watch) bypasses the gate
# so open positions are never left unmanaged.
PROBE_SOURCES = frozenset({"418_ban", "429_retry", "restored", "probe_reject"})
PROBE_MIN_S = 120.0          # bans shorter than this never arm the probe


class RateLimitError(Exception):
    """Raised when a request cannot proceed within the rate budget."""


class WeightedRateLimiter:
    """
    Sliding-window weighted token bucket.

    - `acquire(weight)` blocks until `weight` fits inside the rolling 60s
      window, or returns False if `timeout` elapses first.
    - `trigger_cooldown(seconds)` pauses ALL callers (used on 429/418).
    - `note_server_weight(used)` applies a defensive cooldown when the
      server reports the shared-IP budget is nearly exhausted.

    v5.3: ESCALATING pressure response. On a shared Render IP the neighbors
    (not us) keep `X-MBX-USED-WEIGHT-1M` hot for long stretches. The old
    fixed 10s/30s cooldowns created a poke-loop: every response re-triggered
    the same short cooldown the moment it expired, so the bot crawled and
    the log spammed one WARNING per request. Now consecutive pressure hits
    double the cooldown (10 -> 20 -> 40 -> ... cap 300s) and the warning is
    logged only on transition or meaningful escalation.
    """

    WINDOW_SECONDS = 60.0
    # multiplier applied to the base cooldown after N consecutive hits
    ESCALATION = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    PRESSURE_CAP_S = 300.0      # max cooldown from *header pressure* (not 418)
    PRESSURE_RESET_S = 600.0    # quiet period that resets the streak
    LOW_WEIGHT_PCT = 0.80       # header below this cools the streak down

    def __init__(self, budget_per_min: float = 4500.0, hard_ceiling: int = 6000,
                 priority_reserve: float = 200.0):
        if budget_per_min <= 0:
            budget_per_min = 4500.0
        self.budget = float(budget_per_min)
        self.hard_ceiling = int(hard_ceiling)
        # v5.10: the last PRIORITY_RESERVE weight units of the budget are
        # reserved for PRIORITY requests (position watch, dashboard P&L,
        # pending-entry fills). Bulk traffic (analysis burst, market map)
        # stops at budget - reserve so a full analysis window can never
        # starve the tiny (2w) requests that keep open positions safe.
        self.priority_reserve = max(0.0, float(priority_reserve))
        self._events = deque()  # (monotonic_ts, weight)
        self._lock = threading.Lock()
        self._cooldown_until = 0.0  # monotonic ts
        self._pressure_streak = 0
        self._last_pressure_ts = 0.0
        # v5.29: ban-source visibility + post-ban probe gate state
        self._last_source = "none"
        self._armed_total = 0       # fresh cooldown episodes this process
        self._probe_required = False
        self._probe_claimed = False
        # v5.32: consecutive probe REJECTIONS - drives the probe backoff
        # margin (see trigger_cooldown). Reset on any successful probe.
        self._probe_reject_streak = 0
        # v5.15: restore a ban that was armed by a PREVIOUS process
        self._restore_from_disk()

    # ------------------------------------------------------------------
    def _persist_state(self, seconds: float) -> None:
        """Write the cooldown deadline to disk (epoch-based, atomic)."""
        try:
            path = STATE_FILE
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "cooldown_until_epoch": time.time() + float(seconds),
                "armed_at_epoch": time.time(),
                "last_seconds": round(float(seconds), 1),
                "pid": os.getpid(),
            }
            tmp = path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            os.replace(tmp, path)
        except Exception as e:  # persistence is best-effort, never fatal
            log.debug(f"Rate-state persist failed: {e}")

    def _restore_from_disk(self) -> None:
        """v5.15: re-arm a cooldown armed by a PREVIOUS process (if still live).

        The pid guard keeps same-process semantics intact: within ONE
        process the in-memory state is authoritative (tests and any code
        that builds throwaway limiter instances are unaffected); only a
        genuinely NEW process (Render deploy, crash restart) inherits the
        ban the old process armed.
        """
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if int(payload.get("pid", -1)) == os.getpid():
                return  # same process - in-memory state is authoritative
            until_epoch = float(payload.get("cooldown_until_epoch", 0.0))
        except Exception:
            return  # missing/corrupt file - nothing to restore
        remaining = until_epoch - time.time()
        if remaining <= 1.0:
            try:
                os.remove(STATE_FILE)
            except OSError:
                pass
            return
        self._cooldown_until = time.monotonic() + remaining
        # v5.29: a restored ban is by definition a real server ban - the
        # freshly restarted process must re-probe before releasing traffic.
        self._last_source = "restored"
        self._probe_required = True
        log.warning(
            f"[yellow]Rate limiter[/] restored cooldown from disk: "
            f"{remaining:.0f}s left (previous process armed a 429/418 "
            f"backoff - NOT poking the banned IP)"
        )

    # ------------------------------------------------------------------
    def _prune(self, now: float) -> float:
        """Drop events older than the window. Returns current used weight."""
        horizon = now - self.WINDOW_SECONDS
        while self._events and self._events[0][0] <= horizon:
            self._events.popleft()
        return sum(w for _, w in self._events)

    # ------------------------------------------------------------------
    def acquire(self, weight: float, timeout: float = 90.0,
                priority: bool = False) -> bool:
        """
        Reserve `weight` units. Blocks until they fit in the rolling window.
        Returns True when reserved, False if `timeout` expires first.

        v5.10: `priority=True` (position watch / dashboard / pending fills)
        may use the FULL budget; normal (bulk) traffic is capped at
        budget - priority_reserve so monitoring requests always fit.

        v5.29: while a post-ban probe is pending, bulk traffic is refused
        FAST (False immediately, no 90s wait) so a just-expired ban is not
        re-poked by a thundering herd. Priority traffic still passes - it is
        tiny, and a successful priority request clears the probe.
        """
        weight = max(1.0, float(weight))
        # v5.29: probe gate - fail fast, the caller aborts its tick cleanly
        if self._probe_required and not priority:
            return False
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            now = time.monotonic()
            if now < self._cooldown_until:
                wait = self._cooldown_until - now
                if deadline - now < wait and wait > timeout:
                    return False
                time.sleep(min(wait, 2.0))
                continue

            with self._lock:
                now = time.monotonic()
                used = self._prune(now)
                # v5.10: priority traffic may use the full budget; bulk
                # traffic must leave the priority reserve untouched.
                effective = self.budget if priority else \
                    max(1.0, self.budget - self.priority_reserve)
                if used + weight <= effective:
                    self._events.append((now, weight))
                    return True
                # Weight doesn't fit: figure out how long until enough old
                # events expire to free the required room.
                needed = used + weight - effective
                wait = 0.0
                acc = 0.0
                for ts, w in self._events:  # oldest first
                    acc += w
                    wait = ts + self.WINDOW_SECONDS - now
                    if acc >= needed:
                        break
                wait = max(wait, 0.25)

            if time.monotonic() + wait > deadline:
                return False
            time.sleep(min(wait, 2.0))

    # ------------------------------------------------------------------
    def trigger_cooldown(self, seconds: float,
                         source: str = "unknown") -> None:
        """Pause all outbound requests for `seconds` (429/418 backoff).

        v5.2: cap raised 120s -> 3600s so an IP auto-ban (418) can back off
        hard. Regular 429 paths still pass <= 120s from the client side.
        v5.3: log only on transition into cooldown or a meaningful escalation
        (>= 30s added) - repeated 10/20s re-arms stay silent.

        v5.29: `source` records WHY the cooldown was armed ("418_ban",
        "429_retry", "header_pressure", "restored", "probe_reject") for
        /api/health visibility, and a real server ban of PROBE_MIN_S or
        longer arms the post-ban probe gate. Header-pressure cooldowns
        (<= 300s, neighbor-driven) never arm the probe.

        v5.32: PROBE BACKOFF. source="probe_reject" means the weight-1
        verification probe itself was answered 418 - one more poke into a
        still-banned IP, and poking at the exact Retry-After expiry again
        and again reads as a repeat offense (observed escalation
        1892s -> 2701s -> 3121s). Consecutive rejections now add a growing
        margin ON TOP of the server value (base 300s, x2 per consecutive
        rejection, cap 3600s; PROBE_BACKOFF_BASE_S / PROBE_BACKOFF_MAX_S)
        so our own probe contribution to the herd's ban shrinks the longer
        it lasts. finish_probe(True) resets the streak - a VERIFIED clean
        IP starts the next ban cycle from the base margin.
        """
        # v5.15: cap raised 3600 -> 86400. Binance 418 Retry-After values can
        # exceed 1h for repeat offenders; re-poking after our (capped) cooldown
        # only EXTENDED the ban. Call sites pre-cap their own semantics
        # (429 keeps <= 3600, pressure keeps <= 300); this is a sanity bound.
        # v5.32: the backoff margin is added BEFORE this cap so the total
        # armed cooldown (server value + margin) stays inside the bound.
        seconds = float(seconds)
        now = time.monotonic()
        with self._lock:
            if source == "probe_reject":
                self._probe_reject_streak += 1
                base = float(getattr(_settings, "PROBE_BACKOFF_BASE_S", 300.0))
                cap = float(getattr(_settings, "PROBE_BACKOFF_MAX_S", 3600.0))
                if base > 0.0:
                    margin = min(base * (2 ** (self._probe_reject_streak - 1)),
                                 cap)
                    seconds += margin
            seconds = max(1.0, min(seconds, COOLDOWN_HARD_CAP_S))
            was_in_cooldown = now < self._cooldown_until
            prev_remaining = max(0.0, self._cooldown_until - now) if was_in_cooldown else 0.0
            until = now + seconds
            extended = False
            if until > self._cooldown_until:
                self._cooldown_until = until
                extended = True
            if extended and not was_in_cooldown:
                self._armed_total += 1
            if source and source != "unknown":
                self._last_source = source
            # v5.29: arm the probe for real server bans (not pressure blips)
            if (source in PROBE_SOURCES and seconds >= PROBE_MIN_S
                    and extended):
                self._probe_required = True
                self._probe_claimed = False  # allow a fresh probe after expiry
        added = seconds - prev_remaining
        if extended and (not was_in_cooldown or added >= 30.0):
            log.warning(
                f"[yellow]Rate limiter[/] global cooldown {seconds:.0f}s "
                f"(source: {self._last_source}; server 429/418 or "
                f"shared-IP pressure)"
            )
        # v5.15: persist ONLY real extensions so a restart during a short
        # pressure blip does not inherit a stale ban.
        try:
            min_s = float(getattr(_settings, "RATE_STATE_MIN_S", 60.0))
        except Exception:
            min_s = 60.0
        if extended and seconds >= min_s:
            self._persist_state(seconds)

    def pressure_streak(self) -> int:
        """Consecutive shared-IP pressure hits (for /api/health visibility)."""
        with self._lock:
            return self._pressure_streak

    def _register_pressure(self, base_seconds: float) -> float:
        """Record a pressure hit and return the ESCALATED cooldown seconds."""
        now = time.monotonic()
        with self._lock:
            if now - self._last_pressure_ts > self.PRESSURE_RESET_S:
                self._pressure_streak = 0
            self._pressure_streak += 1
            self._last_pressure_ts = now
            idx = min(self._pressure_streak - 1, len(self.ESCALATION) - 1)
            mult = self.ESCALATION[idx]
        return min(base_seconds * mult, self.PRESSURE_CAP_S)

    def _register_quiet(self) -> None:
        """A healthy header - cool the escalation streak down."""
        with self._lock:
            self._pressure_streak = 0

    def in_cooldown(self) -> bool:
        return time.monotonic() < self._cooldown_until

    def cooldown_remaining(self) -> float:
        """Seconds left in the current cooldown (0.0 if none)."""
        return max(0.0, self._cooldown_until - time.monotonic())

    # ------------------------------------------------------------------
    # v5.29: post-ban probe gate + ban-source visibility
    # ------------------------------------------------------------------
    def needs_probe(self) -> bool:
        """True when a long ban expired but the IP is not yet verified."""
        return self._probe_required and not self.in_cooldown()

    def probe_pending(self) -> bool:
        """Alias of needs_probe() for dashboard readability."""
        return self.needs_probe()

    def begin_probe(self) -> bool:
        """Atomically claim the right to send the verification probe.

        Exactly ONE caller may probe at a time; losers fail fast and retry
        on their next scheduled tick.
        """
        with self._lock:
            if not self._probe_required or self._probe_claimed:
                return False
            if self.in_cooldown():
                return False
            self._probe_claimed = True
            return True

    def finish_probe(self, success: bool) -> None:
        """Resolve the in-flight probe.

        success=True: the IP answered normally - release the herd.
        success=False: the probe was rejected/errored (the client has
        already re-armed a cooldown, v5.32: with a growing backoff margin
        on top) - free the claim so the next tick can try again after the
        new cooldown expires.
        """
        with self._lock:
            self._probe_claimed = False
            if success:
                self._probe_required = False
                self._probe_reject_streak = 0  # v5.32: verified clean -> fresh backoff

    def last_source(self) -> str:
        """Why the last cooldown was armed (for /api/health)."""
        with self._lock:
            return self._last_source

    def armed_total(self) -> int:
        """Fresh cooldown episodes armed by THIS process."""
        with self._lock:
            return self._armed_total

    def probe_reject_streak(self) -> int:
        """v5.32: consecutive probe rejections (backoff depth, /api/health).

        0 = last probe was OK / no rejections yet; rising = the shared IP
        keeps answering 418 and we probe at Retry-After + a growing margin
        (300s x2 per rejection, cap 3600s) instead of poking at expiry."""
        with self._lock:
            return self._probe_reject_streak

    # ------------------------------------------------------------------
    def note_server_weight(self, used: int) -> None:
        """
        Feed back the `X-MBX-USED-WEIGHT-1M` header. If the SHARED IP is
        already near Binance's ceiling, pause even if our own accounting
        is small (neighbors on the same egress IP count too).

        v5.3: consecutive hits ESCALATE (base x2 each time, cap 300s) so a
        hot neighbor stops being poked every few seconds; a healthy header
        (< 80% of ceiling) resets the escalation.
        """
        try:
            used = int(used)
        except (TypeError, ValueError):
            return
        if used <= 0:
            return
        if used >= int(self.hard_ceiling * 0.95):
            self.trigger_cooldown(self._register_pressure(30.0),
                                  source="header_pressure")
        elif used >= int(self.hard_ceiling * 0.85):
            self.trigger_cooldown(self._register_pressure(10.0),
                                  source="header_pressure")
        elif used < int(self.hard_ceiling * self.LOW_WEIGHT_PCT):
            self._register_quiet()

    # ------------------------------------------------------------------
    def used_weight(self) -> float:
        """Current locally-accounted usage inside the window (for tests)."""
        with self._lock:
            return self._prune(time.monotonic())


# Singleton — budget 4500/min = 75% of Binance's 6000 (headroom for shared IP)
from config.settings import settings as _settings

rate_limiter = WeightedRateLimiter(
    budget_per_min=float(_settings.RATE_LIMIT_BUDGET_PER_MIN),
    priority_reserve=float(getattr(_settings, "RATE_PRIORITY_RESERVE", 200.0)),
)
