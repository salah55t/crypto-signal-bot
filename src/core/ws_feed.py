"""WebSocket kline feed - eliminates REST weight for candle polling (v5.3).

Production incident (2026-09-24): the Render shared egress IP sits near
Binance's 6000 weight/min ceiling from NEIGHBOR traffic; our own REST
polling (analyzer klines + bottom-scanner klines = 2x the same 150 symbols
every cycle, ~600+ weight) collided with it and produced endless
429 / header-pressure cooldowns.

Key fact: Binance WebSocket streams do NOT count against the REST
request-weight budget. One combined-stream connection keeps a live 1h
candle cache for the whole universe:

  - SEED   : the first REST get_candles() call ingests into the cache, so
             deployment costs zero extra weight (cycle 1 seeds as usual,
             cycle 2+ reads free).
  - LIVE   : <symbol>@kline_1h events update the in-progress bar in place
             and append new bars as they open (spot pushes every ~2s).
  - SAFE   : get_cached() returns None when the feed is disabled/stale/short
             -> data_fetcher falls back to REST exactly as before. A reconnect
             gap longer than 2 minutes reseeds everything via REST once, so a
             closed bar can never be silently missing (a 1h bar can only be
             missed inside a >2 min gap; the 15-min freshness TTL plus
             reseeding make that impossible).

All public reads are thread-safe; the connection runs in a daemon thread
with exponential reconnect backoff.
"""
import threading
import time
from collections import deque
from typing import Dict, Optional

from config.settings import settings
from src.utils.logger import log

# raw kline row layout (same as data_fetcher.klines_to_df input)
_ROW_LEN = 12


def _kline_event_to_row(k: Dict) -> list:
    """Convert a Binance kline event payload to a raw kline row."""
    return [
        k.get("t"),                     # open_time
        k.get("o"), k.get("h"), k.get("l"), k.get("c"), k.get("v"),
        k.get("T"),                     # close_time
        k.get("q"),                     # quote_volume
        k.get("n"),                     # trades
        k.get("V"),                     # taker_buy_base
        k.get("Q"),                     # taker_buy_quote
        "ws",                           # ignore
    ]


class WSKlineFeed:
    """Live candle cache fed by one combined-market WebSocket stream.

    v5.5: multi-interval. The feed subscribes to every interval listed in
    settings.WS_INTERVALS (strategy TFs + 1h) and keys its cache by
    "SYMBOL|interval", so a 4h strategy and the 1h market map both read
    free without colliding.
    """

    MAX_BARS = 400  # headroom above CANDLE_LIMIT=300 (v5.7, EMA200 warmup) and map lookback=168

    def __init__(self):
        self._bars: Dict[str, deque] = {}   # keyed "SYMBOL|interval"
        self._last_event: Dict[str, float] = {}  # keyed "SYMBOL|interval"
        self._lock = threading.RLock()
        self._started = False
        self._stop = False
        self._ws = None
        self._thread: Optional[threading.Thread] = None
        self._universe: list = []       # desired symbols (UPPERCASE)
        self._connected = False
        self._connected_since = 0.0
        self._last_disconnect_gap = 0.0
        self._needs_reseed = False      # True after a long reconnect gap
        self._reconnect_delay = 5.0
        self._last_msg_ts = 0.0
        self._msg_count = 0

    # ------------------------------------------------------------------
    # cache keying
    # ------------------------------------------------------------------
    @staticmethod
    def _key(symbol: str, interval: str) -> str:
        """Per-(symbol, interval) cache key - intervals must never mix."""
        return f"{symbol.upper().strip()}|{interval}"

    @staticmethod
    def _intervals() -> list:
        """Intervals to subscribe (fallback when WS_INTERVALS is empty)."""
        ivs = list(getattr(settings, "WS_INTERVALS", []) or [])
        return ivs or ["1h"]

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def update_universe(self, symbols: list) -> None:
        """Set/refresh the subscribed symbols; (re)connects when changed."""
        wanted = sorted({s.upper().strip() for s in symbols if s})
        with self._lock:
            changed = wanted != self._universe
            self._universe = wanted
        if changed and self._started:
            log.info(f"[cyan]WS feed[/] universe changed ({len(wanted)} symbols) - reconnecting")
            self._needs_reseed = True
            self._restart()

    def start(self) -> None:
        """Start the feed (no-op when disabled or already running)."""
        if not settings.USE_WS_FEED:
            log.info("[cyan]WS feed[/] disabled (USE_WS_FEED=false)")
            return
        if self._started:
            return
        if not self._universe:
            log.warning("[yellow]WS feed[/] started with empty universe - "
                        "waiting for update_universe()")
        self._started = True
        self._stop = False
        self._thread = threading.Thread(
            target=self._run_loop, name="ws-kline-feed", daemon=True)
        self._thread.start()
        log.info(f"[green]WS kline feed started[/] for {len(self._universe)} symbols "
                 f"x {self._intervals()} ({settings.WS_ENDPOINT})")

    def stop(self) -> None:
        self._stop = True
        self._started = False
        try:
            if self._ws is not None:
                self._ws.close()
        except Exception:
            pass

    def _restart(self) -> None:
        """Force a reconnect with the current universe (background-safe)."""
        try:
            if self._ws is not None:
                self._ws.close()
        except Exception:
            pass

    def _run_loop(self):
        """Connect -> run -> reconnect with backoff until stopped."""
        import websocket  # websocket-client
        while not self._stop:
            streams = []
            with self._lock:
                for iv in self._intervals():
                    streams.extend(
                        f"{s.lower()}@kline_{iv}" for s in self._universe)
            if not streams:
                time.sleep(5.0)
                continue
            url = f"{settings.WS_ENDPOINT}?streams={'/'.join(streams)}"
            try:
                self._ws = websocket.WebSocketApp(
                    url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                )
                # run_forever blocks; ping kept alive by Binance server pings
                self._ws.run_forever(ping_interval=60, ping_timeout=10)
            except Exception as e:
                log.warning(f"[yellow]WS feed[/] connection error: {e}")
            if self._stop:
                break
            time.sleep(self._reconnect_delay)
            self._reconnect_delay = min(self._reconnect_delay * 2, 60.0)

    # ------------------------------------------------------------------
    # websocket callbacks
    # ------------------------------------------------------------------
    def _on_open(self, ws=None):
        now = time.monotonic()
        with self._lock:
            gap = now - (self._last_event_ts_max() or now)
            self._connected = True
            self._connected_since = now
            self._reconnect_delay = 5.0
            # a gap > 2 min could have swallowed an hourly bar close -> reseed
            self._needs_reseed = gap > 120.0
        log.info(f"[green]WS feed connected[/] (gap {gap:.0f}s, reseed={self._needs_reseed})")
        if self._needs_reseed:
            threading.Thread(target=self._reseed_all, daemon=True).start()

    def _on_message(self, ws=None, message: str = ""):
        try:
            import json
            evt = json.loads(message)
            data = evt.get("data") or evt  # combined vs raw stream
            if not isinstance(data, dict) or data.get("e") != "kline":
                return
            k = data.get("k") or {}
            sym = data.get("s")
            interval = k.get("i") or "1h"
            if not sym or not k:
                return
            row = _kline_event_to_row(k)
            now = time.monotonic()
            key = self._key(sym, interval)
            with self._lock:
                bars = self._bars.get(key)
                if bars is None:
                    bars = deque(maxlen=self.MAX_BARS)
                    self._bars[key] = bars
                if bars and bars[-1][0] == row[0]:
                    bars[-1] = row            # in-progress bar update
                elif not bars or row[0] > bars[-1][0]:
                    bars.append(row)          # a new bar opened
                    while len(bars) > self.MAX_BARS:
                        bars.popleft()
                self._last_event[key] = now
            self._last_msg_ts = now
            self._msg_count += 1
        except Exception:
            pass  # never let a malformed frame kill the feed

    def _on_error(self, ws=None, error=None):
        log.debug(f"WS feed error: {error}")

    def _on_close(self, ws=None, code=None, msg=None):
        with self._lock:
            self._connected = False
        log.warning(f"[yellow]WS feed disconnected[/] (code={code}) - reconnecting")

    # ------------------------------------------------------------------
    # reseeding after long gaps
    # ------------------------------------------------------------------
    def _last_event_ts_max(self) -> Optional[float]:
        if not self._last_event:
            return None
        return max(self._last_event.values())

    def _reseed_all(self):
        """REST-refetch every cached series once (after long reconnect gaps)."""
        try:
            from src.core.data_fetcher import DataFetcher  # lazy: avoid circulars
            with self._lock:
                keys = list(self._bars.keys())
            log.info(f"[cyan]WS feed[/] reseeding {len(keys)} series via REST after gap")
            for key in keys:
                if self._stop:
                    return
                sym, _, interval = key.rpartition("|")
                if not sym:
                    continue
                try:
                    # v5.12: a hard 429/418 ban dooms every REST reseed call -
                    # pause the reseed (flag stays True) and let the next
                    # reconnect/next cycle finish it when the ban lifts.
                    from src.core.rate_limiter import rate_limiter
                    if rate_limiter.cooldown_remaining() > 130.0:
                        log.warning(
                            f"[yellow]WS feed reseed paused[/] - rate ban "
                            f"active ({rate_limiter.cooldown_remaining():.0f}s left)"
                        )
                        return
                    # bypass the cache read - reseed must hit REST directly
                    df = DataFetcher._get_candles_rest(sym, interval, 200)
                    self.ingest(sym, df, interval=interval)
                except Exception:
                    continue
            with self._lock:
                self._needs_reseed = False
            log.info("[green]WS feed reseed complete[/]")
        except Exception as e:
            log.warning(f"[yellow]WS feed reseed failed:[/] {e}")

    # ------------------------------------------------------------------
    # public data access
    # ------------------------------------------------------------------
    def ingest(self, symbol: str, df, interval: str = "1h") -> None:
        """Populate/refresh one (symbol, interval) cache from a REST fetch.

        Called by data_fetcher.get_candles so the first REST cycle after a
        deploy seeds the cache with ZERO extra API weight.
        """
        # klines_to_df sets open_time as the INDEX (not a column)
        if df is None or df.empty or getattr(df.index, "name", None) != "open_time":
            return
        try:
            rows = []
            for ts, r in df.tail(self.MAX_BARS).iterrows():
                rows.append([
                    int(ts.timestamp() * 1000),
                    r.get("open"), r.get("high"), r.get("low"), r.get("close"),
                    r.get("volume"), r.get("close_time"), r.get("quote_volume"),
                    r.get("trades"), r.get("taker_buy_base"),
                    r.get("taker_buy_quote"), "rest",
                ])
            now = time.monotonic()
            key = self._key(symbol, interval)
            with self._lock:
                bars = self._bars.get(key)
                if bars is None:
                    bars = deque(maxlen=self.MAX_BARS)
                    self._bars[key] = bars
                bars.clear()
                bars.extend(rows)
                self._last_event[key] = now
        except Exception as e:
            log.debug(f"WS ingest failed for {symbol} {interval}: {e}")

    def fresh(self, symbol: str, min_bars: int = 60,
              interval: str = "1h") -> bool:
        """True when the cache has a live-enough series for the key."""
        if not settings.USE_WS_FEED or not self._started:
            return False
        now = time.monotonic()
        key = self._key(symbol, interval)
        with self._lock:
            bars = self._bars.get(key)
            if not bars or len(bars) < min_bars:
                return False
            last = self._last_event.get(key, 0.0)
        ttl = settings.WS_FRESH_TTL_MIN * 60.0
        return (now - last) <= ttl

    def get_cached(self, symbol: str, limit: int = 200,
                   interval: str = "1h", allow_stale: bool = False,
                   max_age_s: Optional[float] = None):
        """Return a candle DataFrame from the live cache, or None.

        None means "use REST" - every caller must degrade gracefully.

        v5.12: `allow_stale=True` serves the series even when the 15-min
        freshness TTL has lapsed (bounded by `max_age_s`, default 6h).
        Used when a Binance REST ban makes refetching impossible: analyzing
        a few-hours-old 4h series beats aborting the whole cycle. WS events
        keep flowing during REST bans (different service), so the cache is
        often still live anyway.
        """
        from src.core.data_fetcher import DataFetcher  # lazy: avoid circulars
        min_bars = min(limit, 60)
        if not settings.USE_WS_FEED or not self._started:
            return None
        with self._lock:
            bars = self._bars.get(self._key(symbol, interval))
            n = len(bars) if bars else 0
            last = self._last_event.get(self._key(symbol, interval), 0.0)
        if n < min_bars:
            return None
        if not allow_stale:
            if not self.fresh(symbol, min_bars=min_bars, interval=interval):
                return None
        else:
            now = time.monotonic()
            cap = max_age_s if max_age_s is not None else 6 * 3600.0
            if (now - last) > cap:
                return None
        with self._lock:
            bars = list(self._bars.get(self._key(symbol, interval)) or [])
        if len(bars) < limit:
            return None
        try:
            return DataFetcher.klines_to_df(bars[-limit:])
        except Exception:
            return None

    def status(self) -> Dict:
        """Dashboard-facing health snapshot."""
        now = time.monotonic()
        with self._lock:
            intervals = {}
            for key in self._bars:
                iv = key.rpartition("|")[2]
                intervals[iv] = intervals.get(iv, 0) + 1
            return {
                "enabled": bool(settings.USE_WS_FEED),
                "started": self._started,
                "connected": self._connected,
                "universe": len(self._universe),
                "intervals": self._intervals(),
                "cached_symbols": len(self._bars),
                "cached_per_interval": intervals,
                "needs_reseed": self._needs_reseed,
                "last_msg_age_s": round(now - self._last_msg_ts, 1) if self._last_msg_ts else None,
                "messages": self._msg_count,
            }


# Singleton
ws_feed = WSKlineFeed()
