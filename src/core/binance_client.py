"""Binance API client - supports both authenticated and public-only modes.

v5: every outbound request is gated by a weighted sliding-window rate limiter
(Binance allows 6000 weight/min per IP - and Render egress IPs are SHARED).
HTTP 429 responses feed `Retry-After` back into the limiter as a global
cooldown instead of triggering an instant retry storm.

v5.29: POST-BAN PROBE. When a long 429/418 cooldown expires, the first
request verifies the IP with a weight-1 /time probe before the herd (WS
seeder + scalp tick + analysis cycle) is released - a still-banned IP
answers 418 to the tiny probe, the limiter re-arms, and the bulk tick
aborts cleanly instead of EXTENDING the ban (the documented 598s -> 1892s
escalation loop). Also: exchangeInfo (weight 20, multi-MB) is cached in
process for 6h - per-order filter lookups stop paying weight 20 each.
"""
import hmac
import hashlib
import json
import threading
import time
from collections import deque
import urllib.parse
import requests
from typing import Dict, Any, Optional, List
from config.settings import settings
from src.utils.logger import log
from src.core.rate_limiter import rate_limiter, RateLimitError

# Static request-weight map (Binance spot docs).
# Dynamic endpoints (depth/ticker) are computed from params when needed.
_ENDPOINT_WEIGHTS = {
    "/api/v3/ping": 1,
    "/api/v3/time": 1,
    "/api/v3/exchangeInfo": 20,
    "/api/v3/klines": 2,
    "/api/v3/avgPrice": 2,
    "/api/v3/trades": 25,
    "/api/v3/account": 20,
    "/api/v3/order": 1,          # POST place / DELETE cancel
    "/api/v3/order/oco": 1,
}

# v5.29: the post-ban verification probe is always a weight-1 /time request
PROBE_WEIGHT = 1.0


def _endpoint_weight(path: str, params: Dict[str, Any]) -> float:
    """Best-effort request weight for the endpoint + params."""
    if path == "/api/v3/klines":
        # weight scales with the limit: [1,100]->1, (100,500]->2,
        # (500,1000]->5, (1000,1500]->10 (Binance spot docs)
        try:
            lim = int(params.get("limit", 500))
        except (TypeError, ValueError):
            lim = 500
        if lim <= 100:
            return 1
        if lim <= 500:
            return 2
        if lim <= 1000:
            return 5
        return 10
    if path == "/api/v3/depth":
        try:
            lim = int(params.get("limit", 20))
        except (TypeError, ValueError):
            lim = 20
        return 5 if lim <= 100 else 10
    if path == "/api/v3/ticker/24hr":
        # full-market call is weight 80 (!) - single symbol is 2
        return 2 if params.get("symbol") else 80
    if path == "/api/v3/ticker/price":
        syms = params.get("symbols")
        if syms:
            try:
                n = len(json.loads(syms)) if isinstance(syms, str) else len(syms)
            except Exception:
                n = 4
            return 2 if n <= 100 else 4
        return 4  # full price list
    if path == "/api/v3/openOrders":
        return 6
    return _ENDPOINT_WEIGHTS.get(path, 2)


class BinanceClient:
    """
    Lightweight REST client for Binance Spot API.
    Uses public endpoints when no API keys are provided.
    """

    def __init__(self):
        self.base_url = settings.BINANCE_BASE_URL  # for public data (works everywhere)
        self.signed_url = settings.BINANCE_SIGNED_URL  # for private endpoints (place orders)
        self.api_key = settings.BINANCE_API_KEY
        self.api_secret = settings.BINANCE_API_SECRET
        self.session = requests.Session()
        # v5.30: optional egress escape hatch. Render free-tier egress IPs
        # are SHARED with other services' traffic ("the herd") - when the
        # herd alone keeps triggering 418 bans, routing REST through a
        # proxy with a dedicated IP ends the ban cycle permanently. Empty
        # default = direct connection; the WS feed is unaffected either way.
        # v5.33: the proxy is now a MANAGED ROUTE (the "phone bridge") with
        # per-request health: if the phone (Tailscale exit-node) is asleep
        # or offline, transport-level failures transparently fall back to
        # DIRECT egress for BINANCE_PROXY_RETRY_S, then one in-band re-probe
        # flips the route back when the phone returns. Ban responses
        # (418/429) are NOT transport errors and NEVER flip the route - a
        # banned phone IP must not poison the shared one.
        self._proxy_url = getattr(settings, "BINANCE_PROXY_URL", "")
        self._proxy_healthy = True
        self._proxy_retry_at = 0.0
        self._proxy_fail_reason: Optional[str] = None
        self._proxy_state_lock = threading.Lock()
        self._proxy_retry_s = float(
            getattr(settings, "BINANCE_PROXY_RETRY_S", 300.0))
        # v5.33.1: tunable probe timeout - a COLD userspace-WireGuard first
        # handshake (often via DERP relays across continents) can need well
        # over the historical hard-coded 6s, which flapped the route exactly
        # at boot when it mattered most.
        self._probe_timeout_s = float(
            getattr(settings, "BINANCE_PROXY_PROBE_TIMEOUT_S", 6.0))
        self._bg_probe_thread: Optional[threading.Thread] = None
        if self._proxy_url:
            # never log the URL (may embed credentials) - state its presence
            log.info(
                f"[cyan]BinanceClient[/] REST egress routed via managed proxy "
                f"route ({self._proxy_url.split('://', 1)[0]} scheme, "
                f"direct-fallback after {self._proxy_retry_s:.0f}s down)"
            )
        # v5.14: process-lifetime REST request counter (health endpoint
        # visibility for the WS-first weight reduction).
        self.request_count = 0
        # v5.29: exchangeInfo cache - weight 20 + a multi-MB payload per
        # call is pure waste for filter lookups that change ~never.
        self._xinfo: Optional[Dict[str, Any]] = None
        self._xinfo_ts = 0.0
        self._xinfo_lock = threading.Lock()
        self.XINFO_TTL_S = 6 * 3600.0
        # v5.31: per-path REST spend ledger (every request that consumed
        # budget). /api/health exposes the trailing hour so a path that
        # keeps spending is by definition the next ban-risk to fix.
        self._spend: deque = deque()
        self._spend_lock = threading.Lock()
        self._SPEND_CAP = 4000  # bound memory; an hour is a few hundred
        if self.api_key:
            self.session.headers.update({"X-MBX-APIKEY": self.api_key})
        log.info(
            f"[cyan]BinanceClient[/] initialized - "
            f"Mode: {'AUTHENTICATED' if not settings.USE_PUBLIC_ONLY else 'PUBLIC-ONLY'} | "
            f"Public URL: {self.base_url} | "
            f"Signed URL: {self.signed_url}"
        )
        # v5.33.1: optional background route keeper (gated, off in tests)
        self._start_bg_probe()

    # ------------------------------------------------------------------
    # v5.33: managed proxy route (phone bridge) - health + direct fallback
    # ------------------------------------------------------------------
    def _effective_proxies(self) -> Optional[Dict[str, str]]:
        """Per-request proxy mapping for the CURRENT route.

        Returns None = direct egress. When the proxy is configured but
        marked unhealthy, traffic rides direct until the retry window
        elapses; the first caller past the window runs one in-band
        re-probe (weight-0 /ping through the proxy, 6s timeout) and flips
        the route back on success. The state lock serializes the re-probe
        so concurrent callers cannot stampede an unreachable phone.
        """
        if not self._proxy_url:
            return None
        route = {"http": self._proxy_url, "https": self._proxy_url}
        with self._proxy_state_lock:
            if self._proxy_healthy:
                return route
            if time.monotonic() < self._proxy_retry_at:
                return None
            if self._probe_proxy():
                self._proxy_healthy = True
                self._proxy_fail_reason = None
                self._proxy_retry_at = 0.0
                log.info(
                    "[cyan]BinanceClient[/] proxy route RESTORED - "
                    "egress back via the phone bridge"
                )
                return route
            self._proxy_retry_at = time.monotonic() + self._proxy_retry_s
            return None

    def _probe_proxy(self) -> bool:
        """One weight-0 /ping THROUGH the proxy to test reachability."""
        try:
            response = self.session.get(
                f"{self.base_url}/api/v3/ping",
                timeout=self._probe_timeout_s, proxies={
                    "http": self._proxy_url, "https": self._proxy_url})
            return response.status_code == 200
        except Exception as exc:
            # keep a generous cause slice for /api/health - a 120-char cut
            # hides the difference between 'cold tunnel timeout' and
            # 'listener down', which demand different fixes
            self._proxy_fail_reason = f"probe: {exc}"[:400]
            return False

    def _route(self, method: str, url: str, **kwargs):
        """session.get/post through the managed route, with ONE direct
        retry when the proxy itself is unreachable (transport-level errors
        only). 418/429 ban responses are not transport errors and never
        flip the route - a banned phone IP must not poison the shared one.
        """
        proxies = self._effective_proxies()
        used_proxy = proxies is not None
        try:
            return getattr(self.session, method)(url, proxies=proxies,
                                                 **kwargs)
        except (requests.exceptions.ProxyError,
                requests.exceptions.ConnectionError) as exc:
            if not used_proxy:
                raise
            with self._proxy_state_lock:
                self._proxy_healthy = False
                self._proxy_fail_reason = str(exc)[:400]
                self._proxy_retry_at = (time.monotonic()
                                        + self._proxy_retry_s)
            log.warning(
                "[yellow]BinanceClient[/] proxy route UNREACHABLE - "
                f"falling back to DIRECT egress for "
                f"{self._proxy_retry_s:.0f}s (phone asleep/offline?)"
            )
            return getattr(self.session, method)(url, proxies=None, **kwargs)

    def proxy_state(self) -> Dict[str, Any]:
        """/api/health visibility: which route REST is currently riding."""
        with self._proxy_state_lock:
            if not self._proxy_url:
                return {"configured": False, "using": "direct"}
            return {
                "configured": True,
                "scheme": self._proxy_url.split("://", 1)[0],
                "healthy": self._proxy_healthy,
                "using": "proxy" if self._proxy_healthy else "direct-fallback",
                "retry_in_s": (max(0, round(self._proxy_retry_at
                                            - time.monotonic()))
                               if not self._proxy_healthy else 0),
                "last_fail_reason": self._proxy_fail_reason,
            }

    # ------------------------------------------------------------------
    # v5.33.1: background route keeper - heal the bridge with zero traffic
    # ------------------------------------------------------------------
    def _start_bg_probe(self) -> None:
        """Spawn the route-keeper daemon when enabled.

        Gated by BINANCE_PROXY_BG_PROBE (default off so tests and non-
        bridge deployments run thread-free; scripts/render/start.sh turns
        it on for the phone-bridge deployment).
        """
        if not self._proxy_url or not bool(
                getattr(settings, "BINANCE_PROXY_BG_PROBE", False)):
            return
        self._bg_probe_thread = threading.Thread(
            target=self._bg_probe_loop, name="binance-proxy-probe",
            daemon=True)
        self._bg_probe_thread.start()

    def _bg_probe_loop(self) -> None:
        tick_s = float(
            getattr(settings, "BINANCE_PROXY_BG_PROBE_TICK_S", 60.0))
        # boot warm-up: settle the cold-tunnel question NOW so the first
        # real call either rides a verified warm phone route or parks
        # direct - instead of dying through a half-open tunnel.
        self._bg_probe_tick(force=True)
        while True:
            time.sleep(tick_s)
            self._bg_probe_tick()

    def _bg_probe_tick(self, force: bool = False) -> bool:
        """One conditional re-probe under the state lock; True iff the
        route is (now) healthy.

        Without force: probe ONLY when marked down AND the retry window
        has elapsed - never stampede an unreachable phone, never disturb
        a healthy route. This is what keeps the bridge alive while the
        bot is parked in a zero-REST 418 cooldown, where the in-band lazy
        re-probe of _effective_proxies() never gets a caller.
        """
        with self._proxy_state_lock:
            if not force:
                if self._proxy_healthy:
                    return True
                if time.monotonic() < self._proxy_retry_at:
                    return False
            if self._probe_proxy():
                self._proxy_healthy = True
                self._proxy_fail_reason = None
                self._proxy_retry_at = 0.0
                log.info(
                    "[cyan]BinanceClient[/] proxy route RESTORED (bg) - "
                    "egress back via the phone bridge"
                )
                return True
            self._proxy_healthy = False
            self._proxy_retry_at = time.monotonic() + self._proxy_retry_s
            return False

    # ------------------------------------------------------------------
    # v5.29: post-ban verification probe
    # ------------------------------------------------------------------
    def _send_probe(self) -> bool:
        """One weight-1 /time request to verify a just-expired ban.

        Returns True when the IP answers normally (the limiter's probe
        requirement is cleared and the caller proceeds). Returns False when
        the probe was rejected (429/418 - the limiter has already re-armed a
        cooldown with source probe_reject) or errored - the CALLER raises
        RateLimitError and its tick aborts cleanly.
        """
        try:
            if not rate_limiter.acquire(PROBE_WEIGHT, timeout=10.0,
                                        priority=True):
                rate_limiter.finish_probe(False)
                return False
            self.request_count += 1
            # v5.33: route-consistent - the probe verifies whichever route
            # (phone bridge or direct) is currently active.
            response = self._route(
                "get", f"{self.base_url}/api/v3/time", timeout=10)
            used = response.headers.get("X-MBX-USED-WEIGHT-1M")
            if used:
                rate_limiter.note_server_weight(used)
            if response.status_code in (429, 418):
                try:
                    retry_after = float(response.headers.get("Retry-After", ""))
                except ValueError:
                    retry_after = 0.0
                if response.status_code == 418:
                    cooldown = min(max(retry_after, 900.0), 86400.0)
                    rate_limiter.trigger_cooldown(cooldown,
                                                  source="probe_reject")
                else:
                    rate_limiter.trigger_cooldown(
                        min(retry_after + 2.0 if retry_after > 0 else 30.0,
                            3600.0), source="429_retry")
                rate_limiter.finish_probe(False)
                log.warning(
                    "[yellow]Post-ban probe REJECTED[/] - Binance "
                    f"{response.status_code}; re-armed cooldown, the herd "
                    "stays parked (ban likely EXTENDED - repeat offense)")
                return False
            rate_limiter.finish_probe(True)
            log.info("[green]Post-ban probe OK[/] - IP verified, releasing "
                     "the queued traffic")
            return True
        except RateLimitError:
            rate_limiter.finish_probe(False)
            return False
        except Exception as e:
            # network error - do NOT release the herd into an unverified IP
            rate_limiter.finish_probe(False)
            log.warning(f"[yellow]Post-ban probe errored[/] ({e}) - "
                        "retrying on the next tick")
            return False

    def _sign(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Sign request params with HMAC-SHA256 (required for private endpoints)."""
        if not self.api_secret:
            raise PermissionError("API secret required for signed endpoints")
        params["timestamp"] = int(time.time() * 1000)
        query = urllib.parse.urlencode(params)
        signature = hmac.new(
            self.api_secret.encode("utf-8"),
            query.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        params["signature"] = signature
        return params

    def _get(self, path: str, params: Optional[Dict] = None,
             signed: bool = False, priority: bool = False) -> Dict[str, Any]:
        """Perform a GET request to Binance API (rate-limit aware, v5).

        v5.10: `priority=True` requests (position watch, dashboard P&L,
        pending-entry fills) may consume the reserved tail of the rate
        budget that bulk traffic cannot touch."""
        # Use signed_url for signed requests, base_url for public
        base = self.signed_url if signed else self.base_url
        url = f"{base}{path}"
        params = dict(params or {})
        if signed:
            if settings.USE_PUBLIC_ONLY:
                raise PermissionError(
                    "Signed endpoint called but no API keys configured. "
                    "Set BINANCE_API_KEY and BINANCE_API_SECRET in .env"
                )
            params = self._sign(params)

        # v5: reserve request weight BEFORE firing (blocks when the shared
        # IP is near the ceiling; honours 429 cooldowns globally).
        weight = _endpoint_weight(path, params)
        # v5.29: post-ban probe gate. The first non-priority caller after a
        # long ban verifies the IP with a weight-1 /time request; while the
        # probe is pending every other bulk caller fails fast (their ticks
        # abort cleanly and retry on the next schedule).
        if not priority and rate_limiter.needs_probe() \
                and rate_limiter.begin_probe():
            if not self._send_probe():
                raise RateLimitError(
                    "post-ban probe rejected - IP still banned, request aborted"
                )
        if not rate_limiter.acquire(weight, timeout=90.0, priority=priority):
            raise RateLimitError(
                f"Rate budget exhausted ({weight}w needed); "
                "request aborted to avoid a 429 ban"
            )
        self.request_count += 1  # v5.14: only requests that consumed budget
        with self._spend_lock:
            self._spend.append((time.monotonic(), path, float(weight)))
            while len(self._spend) > self._SPEND_CAP:
                self._spend.popleft()

        try:
            # v5.33: managed route (phone bridge + direct fallback)
            response = self._route("get", url, params=params, timeout=15)
            # Feed the server-reported shared-IP usage back into the limiter
            used = response.headers.get("X-MBX-USED-WEIGHT-1M")
            if used:
                rate_limiter.note_server_weight(used)
            if response.status_code == 429 or response.status_code == 418:
                retry_after = response.headers.get("Retry-After", "")
                try:
                    retry_after = float(retry_after)
                except ValueError:
                    retry_after = 0.0
                if response.status_code == 418:
                    # v5.2: IP auto-ban - Retry-After can be minutes..hours and
                    # every request sent during the ban can EXTEND it. Back off
                    # hard (>= 15 min) instead of poking it every cycle.
                    # v5.15: honour the server value up to 24h (repeat offenders
                    # get multi-hour bans; the old 1h cap made us re-poke and
                    # EXTEND the ban). Cooldown survives restarts (rate_state).
                    cooldown = min(max(retry_after, 900.0), 86400.0)
                    rate_limiter.trigger_cooldown(cooldown, source="418_ban")
                else:
                    # v5.12: honor Retry-After FULLY. The old 120s cap made us
                    # poke a still-hot shared IP every 2 minutes, and Binance
                    # punishes repeated 429 violations by escalating to a 418
                    # IP auto-ban (production 2026-09-26: "598s ban left"
                    # abort-storm). If the server names a wait, we wait it.
                    if retry_after <= 0:
                        retry_after = 30.0
                    cooldown = min(retry_after + 2.0, 3600.0)
                    rate_limiter.trigger_cooldown(cooldown, source="429_retry")
                raise RateLimitError(
                    f"Binance {response.status_code}: {response.text[:120]}"
                )
            response.raise_for_status()
            data = response.json()
            # v5.29: any successful response proves the IP is clean - a
            # priority caller that slipped past the probe gate clears it.
            if rate_limiter.probe_pending():
                rate_limiter.finish_probe(True)
            return data
        except RateLimitError:
            raise
        except requests.exceptions.HTTPError as e:
            log.error(f"Binance API HTTP error: {e} - URL: {url} - Response: {response.text[:200]}")
            raise
        except requests.exceptions.RequestException as e:
            log.error(f"Binance API request error: {e}")
            raise

    def rest_spend(self, window_s: float = 3600.0) -> Dict[str, Any]:
        """v5.31: per-endpoint REST spend over the trailing window.

        /api/health visibility for the WS-first weight reduction: after
        v5.31 the steady-state spend should sit near zero, so any path
        that keeps reappearing here is by definition the next ban-risk
        to convert to the WS cache. Weight is the limiter budget actual
        requests consumed (pre-fire accounting, so aborted-by-4xx calls
        still show - they cost the same shared-IP goodwill).
        """
        horizon = time.monotonic() - max(1.0, float(window_s))
        with self._spend_lock:
            events = [e for e in self._spend if e[0] > horizon]
            self._spend.clear()
            self._spend.extend(events)
        agg: Dict[str, Dict[str, float]] = {}
        for _, path, w in events:
            slot = agg.setdefault(path, {"calls": 0, "weight": 0.0})
            slot["calls"] += 1
            slot["weight"] += w
        ranked = sorted(agg.items(), key=lambda kv: kv[1]["weight"],
                        reverse=True)
        return {
            "window_s": int(window_s),
            "total_calls": int(sum(v["calls"] for _, v in ranked)),
            "total_weight": int(sum(v["weight"] for _, v in ranked)),
            "paths": {p: {"calls": int(v["calls"]),
                          "weight": int(v["weight"])}
                      for p, v in ranked[:10]},
        }

    def get_tickers_batch(self, symbols: List[str],
                          priority: bool = False) -> Dict[str, Dict]:
        """
        v5: batched /api/v3/ticker/price for selected symbols.
        Weight 2 (<=100 symbols) instead of 80 for the full /ticker/24hr
        call - the previous code fetched ALL tickers just to read 5 prices.
        Returns {symbol: {"price": float, ...}}.
        """
        if not symbols:
            return {}
        if len(symbols) > 100:
            # endpoint caps at 100 symbols per call
            out: Dict[str, Dict] = {}
            for i in range(0, len(symbols), 100):
                out.update(self.get_tickers_batch(symbols[i:i + 100],
                                                  priority=priority))
            return out
        # Binance rejects spaces in the symbols array (code -1100):
        # json.dumps default separator is ", " -> ["A", "B"] is INVALID.
        # separators=(",", ":") produces ["A","B"] as the API requires.
        params = {"symbols": json.dumps(list(symbols), separators=(",", ":"))}
        rows = self._get("/api/v3/ticker/price", params, priority=priority)
        rows = rows if isinstance(rows, list) else [rows]
        return {r["symbol"]: r for r in rows if r.get("symbol")}

    # ============================================
    # PUBLIC ENDPOINTS (no API key needed)
    # ============================================

    def ping(self) -> bool:
        """Test connectivity."""
        try:
            r = self._get("/api/v3/ping")
            return r == {}
        except Exception:
            return False

    def get_server_time(self) -> int:
        """Get Binance server time (ms)."""
        return self._get("/api/v3/time").get("serverTime")

    def get_exchange_info(self, force: bool = False) -> Dict[str, Any]:
        """Get exchange info (all symbols, filters, etc.).

        v5.29: cached in-process for 6h (weight 20 + multi-MB payload per
        call; symbol filters change ~never). `force=True` bypasses the
        cache for callers that genuinely need fresh data.
        """
        now = time.time()
        if (not force and self._xinfo is not None
                and (now - self._xinfo_ts) < self.XINFO_TTL_S):
            return self._xinfo
        with self._xinfo_lock:
            # double-check inside the lock (thundering boot)
            if (not force and self._xinfo is not None
                    and (time.time() - self._xinfo_ts) < self.XINFO_TTL_S):
                return self._xinfo
            info = self._get("/api/v3/exchangeInfo")
            self._xinfo = info
            self._xinfo_ts = time.time()
            return info

    def get_all_tickers(self) -> List[Dict[str, Any]]:
        """Get 24h ticker stats for all symbols (weight 80 - avoid for prices)."""
        return self._get("/api/v3/ticker/24hr")

    def get_all_prices(self, priority: bool = False) -> List[Dict[str, Any]]:
        """v5.10: latest price for ALL symbols via /api/v3/ticker/price.

        Weight 4 for the WHOLE market (vs 80 for /ticker/24hr) - the correct
        fallback when the batched call fails: 20x cheaper and it still
        covers every symbol (rows carry the "price" key).
        """
        rows = self._get("/api/v3/ticker/price", priority=priority)
        return rows if isinstance(rows, list) else [rows]

    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """Get 24h ticker stats for a symbol."""
        return self._get("/api/v3/ticker/24hr", {"symbol": symbol})

    def get_klines(self, symbol: str, interval: str = "1h",
                   limit: int = 200, start_time: Optional[int] = None,
                   end_time: Optional[int] = None) -> List[List]:
        """
        Get historical klines (candles).
        Response: [[openTime, open, high, low, close, volume, closeTime,
                    quoteAssetVolume, trades, takerBuyBase, takerBuyQuote, ignore], ...]
        """
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        return self._get("/api/v3/klines", params)

    def get_order_book(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """Get order book depth."""
        return self._get("/api/v3/depth", {"symbol": symbol, "limit": limit})

    def get_recent_trades(self, symbol: str, limit: int = 50) -> List[Dict]:
        """Get recent trades."""
        return self._get("/api/v3/trades", {"symbol": symbol, "limit": limit})

    def get_avg_price(self, symbol: str) -> Dict[str, Any]:
        """Get current average price."""
        return self._get("/api/v3/avgPrice", {"symbol": symbol})

    # ============================================
    # PRIVATE ENDPOINTS (require API key + secret)
    # ============================================

    def get_account_info(self) -> Dict[str, Any]:
        """Get account information (balances, fees, etc.)."""
        return self._get("/api/v3/account", signed=True)

    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get current open orders."""
        params = {}
        if symbol:
            params["symbol"] = symbol
        return self._get("/api/v3/openOrders", params, signed=True)

    def _post(self, path: str, params: Optional[Dict] = None,
              signed: bool = True) -> Dict[str, Any]:
        """Perform a signed POST request (used for placing orders). v5: rate-limit aware."""
        import time as _t
        # Use signed_url (api.binance.com) for private endpoints
        url = f"{self.signed_url}{path}"
        params = dict(params or {})
        params["timestamp"] = int(_t.time() * 1000)
        if signed:
            if settings.USE_PUBLIC_ONLY:
                raise PermissionError(
                    "Signed endpoint called but no API keys configured."
                )
            params = self._sign(params)

        weight = _endpoint_weight(path, params)
        if not rate_limiter.acquire(weight, timeout=90.0):
            raise RateLimitError(
                f"Rate budget exhausted before POST {path}"
            )
        try:
            # v5.33: managed route (phone bridge + direct fallback)
            response = self._route("post", url, params=params, timeout=15)
            used = response.headers.get("X-MBX-USED-WEIGHT-1M")
            if used:
                rate_limiter.note_server_weight(used)
            if response.status_code in (429, 418):
                retry_after = response.headers.get("Retry-After", "30")
                try:
                    retry_after = float(retry_after)
                except ValueError:
                    retry_after = 30.0
                rate_limiter.trigger_cooldown(min(retry_after + 2, 120),
                                              source="429_retry")
                raise RateLimitError(
                    f"Binance {response.status_code}: {response.text[:120]}"
                )
            response.raise_for_status()
            return response.json()
        except RateLimitError:
            raise
        except requests.exceptions.HTTPError as e:
            log.error(f"Binance POST HTTP error: {e} - Response: {response.text[:300]}")
            raise
        except requests.exceptions.RequestException as e:
            log.error(f"Binance POST request error: {e}")
            raise

    def place_market_buy(self, symbol: str, quote_quantity: float) -> Dict:
        """
        Place a MARKET BUY order using quoteOrderQty (e.g. buy $50 of BTC).
        Returns the order response from Binance.
        """
        if settings.USE_PUBLIC_ONLY:
            raise PermissionError("Cannot place real orders without API keys")
        params = {
            "symbol": symbol,
            "side": "BUY",
            "type": "MARKET",
            "quoteOrderQty": round(quote_quantity, 8),
        }
        log.info(f"[yellow]PLACING REAL MARKET BUY[/] {symbol} quote=${quote_quantity}")
        return self._post("/api/v3/order", params, signed=True)

    def place_limit_buy(self, symbol: str, quantity: float, price: float,
                       time_in_force: str = "GTC") -> Dict:
        """Place a LIMIT BUY order."""
        if settings.USE_PUBLIC_ONLY:
            raise PermissionError("Cannot place real orders without API keys")
        params = {
            "symbol": symbol,
            "side": "BUY",
            "type": "LIMIT",
            "timeInForce": time_in_force,
            "quantity": round(quantity, 8),
            "price": round(price, 8),
        }
        log.info(f"[yellow]PLACING REAL LIMIT BUY[/] {symbol} qty={quantity} @ {price}")
        return self._post("/api/v3/order", params, signed=True)

    def place_oco_sell(self, symbol: str, quantity: float,
                       take_profit_price: float, stop_loss_price: float,
                       stop_limit_price: float) -> Dict:
        """
        Place an OCO (One-Cancels-the-Other) SELL order —
        automatically places both TP and SL. When one triggers, the other is cancelled.
        """
        if settings.USE_PUBLIC_ONLY:
            raise PermissionError("Cannot place real orders without API keys")
        params = {
            "symbol": symbol,
            "side": "SELL",
            "quantity": round(quantity, 8),
            "price": round(take_profit_price, 8),
            "stopPrice": round(stop_loss_price, 8),
            "stopLimitPrice": round(stop_limit_price, 8),
            "stopLimitTimeInForce": "GTC",
            "listOrderSide": "SELL",
        }
        log.info(f"[yellow]PLACING REAL OCO SELL[/] {symbol} qty={quantity} "
                 f"TP={take_profit_price} SL={stop_loss_price}")
        return self._post("/api/v3/order/oco", params, signed=True)

    def cancel_order(self, symbol: str, order_id: int) -> Dict:
        """Cancel an open order."""
        if settings.USE_PUBLIC_ONLY:
            raise PermissionError("Cannot cancel orders without API keys")
        params = {"symbol": symbol, "orderId": order_id}
        log.info(f"[yellow]CANCELLING ORDER[/] {symbol} id={order_id}")
        return self._post("/api/v3/order", params, signed=True)

    def get_account_balances(self) -> Dict[str, float]:
        """Return non-zero balances {asset: amount}."""
        if settings.USE_PUBLIC_ONLY:
            return {}
        info = self.get_account_info()
        return {
            b["asset"]: float(b["free"])
            for b in info.get("balances", [])
            if float(b["free"]) > 0
        }

    def get_symbol_filters(self, symbol: str) -> Dict:
        """Get trading filters for a symbol (LOT_SIZE, PRICE_FILTER, etc.)."""
        info = self.get_exchange_info()
        for s in info.get("symbols", []):
            if s["symbol"] == symbol:
                filters = {f["filterType"]: f for f in s.get("filters", [])}
                return {
                    "base_asset": s.get("baseAsset"),
                    "quote_asset": s.get("quoteAsset"),
                    "lot_size_step": float(filters.get("LOT_SIZE", {}).get("stepSize", 0.00000001)),
                    "lot_size_min": float(filters.get("LOT_SIZE", {}).get("minQty", 0)),
                    "lot_size_max": float(filters.get("LOT_SIZE", {}).get("maxQty", 0)),
                    "min_notional": float(filters.get("MIN_NOTIONAL", {}).get("minNotional", 10)),
                    "tick_size": float(filters.get("PRICE_FILTER", {}).get("tickSize", 0.00000001)),
                }
        return {}

    def round_quantity_to_lot(self, symbol: str, quantity: float) -> float:
        """Round quantity to the symbol's LOT_SIZE step."""
        filters = self.get_symbol_filters(symbol)
        if not filters:
            return quantity
        step = filters["lot_size_step"]
        min_qty = filters["lot_size_min"]
        if quantity < min_qty:
            return 0
        # Round down to nearest step
        import math
        rounded = math.floor(quantity / step) * step
        return round(rounded, 8)

    # ============================================
    # CONVENIENCE METHODS
    # ============================================

    def get_top_usdt_symbols_by_volume(self, limit: int = 50) -> List[str]:
        """Get top USDT spot pairs by 24h quote volume."""
        tickers = self.get_all_tickers()
        usdt_pairs = [
            t for t in tickers
            if t.get("symbol", "").endswith("USDT")
            and float(t.get("quoteVolume", 0)) > 0
        ]
        usdt_pairs.sort(key=lambda t: float(t.get("quoteVolume", 0)), reverse=True)
        return [t["symbol"] for t in usdt_pairs[:limit]]


# Singleton
binance_client = BinanceClient()
