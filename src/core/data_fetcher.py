"""Data fetcher - retrieves and formats market data from Binance.

v5: order book snapshots are cached with a TTL (weight 5 per fetch) and
24h tickers use the batched price endpoint when only prices are needed.
"""
import json
import os
import time
from pathlib import Path

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from config.settings import settings
from src.core.binance_client import binance_client
from src.core.rate_limiter import rate_limiter, RateLimitError
from src.utils.logger import log
from src.utils.helpers import retry_on_failure


class DataFetcher:
    """Fetches OHLCV data and order book snapshots for analysis."""

    # v5.25: "1s" added - the only sub-minute interval Binance spot offers.
    # It is the building block of the micro-scalper: 1s bars are fetched and
    # resampled locally into the strategy's true 15s/30s candles. NOTE: "1s"
    # is deliberately NOT part of settings.WS_INTERVALS (a @kline_1s stream
    # for the whole universe would explode the WS message rate); it always
    # takes the REST path via get_candles().
    INTERVALS = {"1s", "1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h",
                 "6h", "8h", "12h", "1d", "3d", "1w", "1M"}

    def __init__(self):
        # v5: {symbol: (monotonic_ts, order_book_dict)} - cuts order book
        # weight from 5/symbol/cycle to 5/symbol/TTL (30 min default).
        self._ob_cache: Dict[str, tuple] = {}

    @staticmethod
    def klines_to_df(raw_klines: List[List]) -> pd.DataFrame:
        """Convert raw Binance klines to a clean DataFrame.

        v5.24: timestamp columns are normalized ELEMENT-WISE. Producers mix
        types - REST rows carry int-ms (and the cache `ingest()` used to
        carry pandas Timestamps in `close_time`!) while live WS event rows
        carry int-ms - and `pd.to_datetime(..., unit='ms')` RAISES on the
        mixture (production 2026-10-01: every WS-touched series became
        unreadable -> a degraded WS-only cycle served 1 of 92 symbols with
        the failure swallowed into get_cached()'s None). One int-ms value
        next to a Timestamp no longer poisons the column.
        """
        if not raw_klines:
            return pd.DataFrame()
        columns = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ]
        df = pd.DataFrame(raw_klines, columns=columns)
        # Convert timestamps (element-wise int-ms normalization first)
        for col in ("open_time", "close_time"):
            df[col] = df[col].map(
                lambda v: v if isinstance(v, (int, float))
                and not isinstance(v, bool)
                else DataFetcher._ts_to_ms(v))
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        # Cast numerics
        numeric_cols = ["open", "high", "low", "close", "volume",
                        "quote_volume", "trades", "taker_buy_base",
                        "taker_buy_quote"]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        df.set_index("open_time", inplace=True)
        df.drop(columns=["ignore"], inplace=True)
        return df

    @staticmethod
    def _ts_to_ms(v) -> Optional[int]:
        """v5.24: best-effort conversion of any timestamp-ish value to ms."""
        try:
            return int(pd.Timestamp(v).timestamp() * 1000)
        except Exception:
            return None

    @staticmethod
    @retry_on_failure
    def get_candles(symbol: str, interval: str = "1h",
                    limit: int = 200) -> pd.DataFrame:
        """Get historical candles for a symbol as a DataFrame.

        v5.3: for cached intervals the live WebSocket cache is served first
        (zero REST weight - Binance WS streams bypass the request-weight
        budget). v5.5: every interval in settings.WS_INTERVALS is cached
        (strategy TFs + 1h), with per-(symbol, interval) keys.
        Falls back to REST when the feed is disabled/stale/short, and any
        REST fetch re-ingests into the cache so the next cycle reads free.
        """
        if interval not in DataFetcher.INTERVALS:
            raise ValueError(f"Invalid interval '{interval}'. Valid: {DataFetcher.INTERVALS}")
        if interval in (settings.WS_INTERVALS or ["1h"]):
            try:
                from src.core.ws_feed import ws_feed
                cached = ws_feed.get_cached(symbol, limit, interval=interval)
                if cached is not None:
                    return cached
                # v5.12: during a hard 429/418 REST ban, refetching is
                # impossible - serve the slightly stale WS series (WS events
                # keep flowing; REST bans do not touch the WS service).
                # A few-hours-old 4h candle set beats an aborted cycle.
                # v5.14: serve during ANY active cooldown (was > 60s) - short
                # shared-IP pressure pauses must not poke REST either.
                from src.core.rate_limiter import rate_limiter
                if rate_limiter.cooldown_remaining() > 0.0:
                    stale = ws_feed.get_cached(
                        symbol, limit, interval=interval,
                        allow_stale=True, max_age_s=6 * 3600.0,
                    )
                    if stale is not None:
                        log.debug(
                            f"Rate ban active - serving stale WS cache for "
                            f"{symbol} {interval}"
                        )
                        return stale
            except Exception:
                pass  # cache must never break the REST path
        return DataFetcher._get_candles_rest(symbol, interval, limit)

    @staticmethod
    def _get_candles_rest(symbol: str, interval: str = "1h",
                          limit: int = 200) -> pd.DataFrame:
        """REST fetch + cache ingest (bypasses the WS read - used by reseeds).

        v5.12: fast-fail during a hard 429/418 ban (>130s left) WITHOUT
        entering retry_on_failure - during the 2026-09-26 598s ban every
        queued call still walked the acquire->raise->retry path, spamming
        the log and churning CPU for a guaranteed failure. One check here
        protects get_candles, reseeds and any direct REST caller.
        """
        from src.core.rate_limiter import rate_limiter, RateLimitError
        ban_left = rate_limiter.cooldown_remaining()
        if ban_left > 130.0:
            raise RateLimitError(
                f"Binance REST ban active ({ban_left:.0f}s left) - "
                f"klines fetch skipped"
            )
        raw = binance_client.get_klines(symbol, interval, limit=limit)
        df = DataFetcher.klines_to_df(raw)
        if interval in (settings.WS_INTERVALS or ["1h"]) \
                and df is not None and not df.empty:
            try:
                from src.core.ws_feed import ws_feed
                ws_feed.ingest(symbol, df, interval=interval)
            except Exception:
                pass
        return df

    @staticmethod
    @retry_on_failure
    def get_multi_timeframe_candles(symbol: str, intervals: List[str],
                                    limit: int = 200) -> Dict[str, pd.DataFrame]:
        """Fetch candles across multiple timeframes."""
        result = {}
        for tf in intervals:
            try:
                result[tf] = DataFetcher.get_candles(symbol, tf, limit)
            except Exception as e:
                log.warning(f"Failed to fetch {symbol} {tf}: {e}")
        return result

    @staticmethod
    @retry_on_failure
    def get_order_book(symbol: str, limit: int = 20) -> Dict:
        """Get order book with computed metrics (v5: TTL-cached)."""
        return DataFetcher._get_order_book_cached(symbol, limit)

    @classmethod
    def _get_order_book_cached(cls, symbol: str, limit: int) -> Dict:
        """Return a cached snapshot when fresh enough (liquidity moves slowly)."""
        inst = data_fetcher
        ttl = max(1, settings.ORDER_BOOK_TTL_MIN) * 60
        now = time.monotonic()
        cached = inst._ob_cache.get(symbol)
        if cached and (now - cached[0]) < ttl:
            return cached[1]
        ob = binance_client.get_order_book(symbol, limit)
        bids = pd.DataFrame(ob.get("bids", []), columns=["price", "qty"], dtype=float)
        asks = pd.DataFrame(ob.get("asks", []), columns=["price", "qty"], dtype=float)
        result = {
            "symbol": symbol,
            "bids": bids,
            "asks": asks,
            "last_update_id": ob.get("lastUpdateId"),
            "cached": bool(cached),
        }
        inst._ob_cache[symbol] = (now, result)
        return result

    @staticmethod
    @retry_on_failure
    def get_ticker_24h(symbol: str) -> Dict:
        """Get 24h ticker statistics."""
        return binance_client.get_ticker(symbol)

    @staticmethod
    def get_batch_tickers(symbols: List[str],
                          priority: bool = False) -> Dict[str, Dict]:
        """Get latest prices for many symbols.

        v5.10 fallback chain (was: full-market /ticker/24hr = weight 80!):
          1. batched /ticker/price?symbols=[...]  -> weight 2-4
          2. full /ticker/price (ALL symbols)     -> weight 4  (20x cheaper
             than the old 24hr fallback, still covers every symbol)
        Both steps run on the priority lane when requested so position
        monitoring survives a bulk analysis burst.
        """
        if not symbols:
            return {}
        try:
            return binance_client.get_tickers_batch(symbols, priority=priority)
        except RateLimitError:
            # v5.23: budget exhaustion dooms the weight-4 full-market list
            # TOO (it costs MORE than the 2w batch it "falls back" to).
            # Production 2026-10-01: every dashboard poll logged the same
            # failure twice (2w + 4w). Re-raise once; the caller's
            # last-known fallback takes over immediately.
            raise
        except Exception as e:
            log.warning(
                f"Batch ticker fetch failed ({e}) - "
                f"falling back to full price list (weight 4)")
        all_p = binance_client.get_all_prices(priority=True)
        sym_set = set(symbols)
        return {t["symbol"]: t for t in all_p if t["symbol"] in sym_set}

    # ---- v5.10: last-known prices (serves the dashboard when even the
    # fallback cannot run - e.g. shared-IP pressure cooldown) ----
    _LAST_PRICES: Dict[str, tuple] = {}  # symbol -> (time.time(), price)

    # v5.23: disk-backed last-known prices. Render redeploys wipe the
    # in-memory table; the cooldown restored from rate_state.json then
    # blocks every REST price fetch for minutes - open positions opened by
    # the bottom/momentum channels (SPCXBUSDT/CRCLBUSDT 2026-10-01) went
    # price-blind on the dashboard AND in the 1-min SL/TP monitor. The file
    # is written throttled and read lazily on the first memory miss.
    _LAST_PRICES_FILE = Path("data/last_prices.json")
    _LAST_PRICES_LAST_SAVE = 0.0
    _LAST_PRICES_DISK_LOADED = False

    @staticmethod
    def _save_last_prices_disk() -> None:
        """v5.23: throttled (30s) atomic write of last-known prices."""
        now = time.time()
        if now - DataFetcher._LAST_PRICES_LAST_SAVE < 30.0:
            return
        DataFetcher._LAST_PRICES_LAST_SAVE = now
        try:
            path = DataFetcher._LAST_PRICES_FILE
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                sym: {"price": price, "ts": ts}
                for sym, (ts, price) in DataFetcher._LAST_PRICES.items()
            }
            tmp = path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f)
            os.replace(tmp, path)
        except Exception as e:  # persistence is best-effort, never fatal
            log.debug(f"Last-prices disk save failed: {e}")

    @staticmethod
    def _load_last_prices_disk() -> None:
        """v5.23: one-time lazy load; memory entries win (fresher)."""
        DataFetcher._LAST_PRICES_DISK_LOADED = True
        try:
            with open(DataFetcher._LAST_PRICES_FILE, "r", encoding="utf-8") as f:
                payload = json.load(f)
            for sym, row in (payload or {}).items():
                try:
                    ts = float(row.get("ts", 0.0))
                    price = float(row.get("price", 0.0))
                except (TypeError, ValueError, AttributeError):
                    continue
                if price > 0 and ts > 0:
                    DataFetcher._LAST_PRICES.setdefault(sym, (ts, price))
        except FileNotFoundError:
            pass
        except Exception as e:
            log.debug(f"Last-prices disk load failed: {e}")

    @staticmethod
    def get_batch_prices(symbols: List[str],
                         priority: bool = False) -> Dict[str, float]:
        """v5: {symbol: lastPrice} for a symbol list.

        v5.14 WS-FIRST: live prices come from the miniTicker WS streams
        (ZERO REST weight, ~1 update/s per symbol) and REST is contacted
        ONLY for symbols the feed cannot serve. WS keeps flowing during a
        429/418 ban, so position watch / dashboard P&L / pending fills stay
        fully live while REST is banned - the old path fell back to stale
        last-known prices instead.

        v5.23: during ANY active cooldown the REST gap-fill is SKIPPED
        entirely (acquire could only block up to 90s to refuse) - the
        caller combines the partial result with last-known prices. This
        kills the "budget exhausted (2w)" + "(4w)" double failure the
        dashboard produced on every poll during a ban.
        """
        if not symbols:
            return {}
        out: Dict[str, float] = {}
        missing = list(dict.fromkeys(symbols))
        # 1) live WS prices (free)
        try:
            from src.core.ws_feed import ws_feed
            ws_prices = ws_feed.get_live_prices(symbols)
            for sym, price in ws_prices.items():
                out[sym] = price
                DataFetcher._LAST_PRICES[sym] = (time.time(), price)
            missing = [s for s in missing if s not in out]
        except Exception:
            missing = list(dict.fromkeys(symbols))
        # 2) REST only for the gaps (weight 2-4 per batched request)
        if missing:
            if rate_limiter.cooldown_remaining() > 0.0 \
                    or rate_limiter.needs_probe():
                # v5.23: a ban dooms every REST call - return what WS served
                # and let the caller's last-known fallback fill the gaps.
                # v5.30: the PROBE WINDOW (cooldown expired, IP not yet
                # verified) is also no-REST. Priority callers bypass the
                # probe gate inside the client, so without this guard a
                # dashboard poll or a position-watch gap-fill would send
                # real requests into a still-418 IP and EXTEND the ban
                # (the repeat-offense loop: probe 2701s -> poke -> extend).
                # WS streams keep flowing during REST bans, so position
                # watch / dashboard P&L stay live; last-known covers gaps.
                return out
            for sym, t in DataFetcher.get_batch_tickers(
                    missing, priority=priority).items():
                try:
                    price = float(t.get("lastPrice", t.get("price", 0)))
                except (TypeError, ValueError):
                    continue
                if price > 0:
                    out[sym] = price
                    DataFetcher._LAST_PRICES[sym] = (time.time(), price)
        # v5.23: persist fresh prices so a REDEPLOY during a ban can still
        # serve position P&L from disk on its very first polls.
        if out:
            DataFetcher._save_last_prices_disk()
        return out

    @staticmethod
    def get_last_known_prices(symbols: List[str],
                              max_age_s: float = 900.0) -> Dict[str, float]:
        """v5.10: last successfully fetched prices, filtered by freshness.

        v5.23: the first call in a process lazily loads the disk snapshot
        (data/last_prices.json) - a fresh Render process that boots inside
        a restored cooldown can still price open positions immediately.
        """
        if not DataFetcher._LAST_PRICES_DISK_LOADED:
            DataFetcher._load_last_prices_disk()
        now = time.time()
        out = {}
        for sym in set(symbols):
            entry = DataFetcher._LAST_PRICES.get(sym)
            if entry and (now - entry[0]) <= max_age_s:
                out[sym] = entry[1]
        return out


# Singleton
data_fetcher = DataFetcher()
