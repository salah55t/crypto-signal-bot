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
import threading
import time
from collections import deque

from src.utils.logger import log


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

    def __init__(self, budget_per_min: float = 4500.0, hard_ceiling: int = 6000):
        if budget_per_min <= 0:
            budget_per_min = 4500.0
        self.budget = float(budget_per_min)
        self.hard_ceiling = int(hard_ceiling)
        self._events = deque()  # (monotonic_ts, weight)
        self._lock = threading.Lock()
        self._cooldown_until = 0.0  # monotonic ts
        self._pressure_streak = 0
        self._last_pressure_ts = 0.0

    # ------------------------------------------------------------------
    def _prune(self, now: float) -> float:
        """Drop events older than the window. Returns current used weight."""
        horizon = now - self.WINDOW_SECONDS
        while self._events and self._events[0][0] <= horizon:
            self._events.popleft()
        return sum(w for _, w in self._events)

    # ------------------------------------------------------------------
    def acquire(self, weight: float, timeout: float = 90.0) -> bool:
        """
        Reserve `weight` units. Blocks until they fit in the rolling window.
        Returns True when reserved, False if `timeout` expires first.
        """
        weight = max(1.0, float(weight))
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
                if used + weight <= self.budget:
                    self._events.append((now, weight))
                    return True
                # Weight doesn't fit: figure out how long until enough old
                # events expire to free the required room.
                needed = used + weight - self.budget
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
    def trigger_cooldown(self, seconds: float) -> None:
        """Pause all outbound requests for `seconds` (429/418 backoff).

        v5.2: cap raised 120s -> 3600s so an IP auto-ban (418) can back off
        hard. Regular 429 paths still pass <= 120s from the client side.
        v5.3: log only on transition into cooldown or a meaningful escalation
        (>= 30s added) - repeated 10/20s re-arms stay silent.
        """
        seconds = max(1.0, min(float(seconds), 3600.0))
        now = time.monotonic()
        with self._lock:
            was_in_cooldown = now < self._cooldown_until
            prev_remaining = max(0.0, self._cooldown_until - now) if was_in_cooldown else 0.0
            until = now + seconds
            extended = False
            if until > self._cooldown_until:
                self._cooldown_until = until
                extended = True
        added = seconds - prev_remaining
        if extended and (not was_in_cooldown or added >= 30.0):
            log.warning(
                f"[yellow]Rate limiter[/] global cooldown {seconds:.0f}s "
                f"(server 429/418 or shared-IP pressure)"
            )

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
            self.trigger_cooldown(self._register_pressure(30.0))
        elif used >= int(self.hard_ceiling * 0.85):
            self.trigger_cooldown(self._register_pressure(10.0))
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
)
