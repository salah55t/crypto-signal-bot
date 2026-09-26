"""Data fetcher - retrieves and formats market data from Binance.

v5: order book snapshots are cached with a TTL (weight 5 per fetch) and
24h tickers use the batched price endpoint when only prices are needed.
"""
import time
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from config.settings import settings
from src.core.binance_client import binance_client
from src.utils.logger import log
from src.utils.helpers import retry_on_failure


class DataFetcher:
    """Fetches OHLCV data and order book snapshots for analysis."""

    INTERVALS = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h",
                 "6h", "8h", "12h", "1d", "3d", "1w", "1M"}

    def __init__(self):
        # v5: {symbol: (monotonic_ts, order_book_dict)} - cuts order book
        # weight from 5/symbol/cycle to 5/symbol/TTL (30 min default).
        self._ob_cache: Dict[str, tuple] = {}

    @staticmethod
    def klines_to_df(raw_klines: List[List]) -> pd.DataFrame:
        """Convert raw Binance klines to a clean DataFrame."""
        if not raw_klines:
            return pd.DataFrame()
        columns = [
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ]
        df = pd.DataFrame(raw_klines, columns=columns)
        # Convert timestamps
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
        # Cast numerics
        numeric_cols = ["open", "high", "low", "close", "volume",
                        "quote_volume", "trades", "taker_buy_base", "taker_buy_quote"]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        df.set_index("open_time", inplace=True)
        df.drop(columns=["ignore"], inplace=True)
        return df

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
            for sym, t in DataFetcher.get_batch_tickers(
                    missing, priority=priority).items():
                try:
                    price = float(t.get("lastPrice", t.get("price", 0)))
                except (TypeError, ValueError):
                    continue
                if price > 0:
                    out[sym] = price
                    DataFetcher._LAST_PRICES[sym] = (time.time(), price)
        return out

    @staticmethod
    def get_last_known_prices(symbols: List[str],
                              max_age_s: float = 900.0) -> Dict[str, float]:
        """v5.10: last successfully fetched prices, filtered by freshness."""
        now = time.time()
        out = {}
        for sym in set(symbols):
            entry = DataFetcher._LAST_PRICES.get(sym)
            if entry and (now - entry[0]) <= max_age_s:
                out[sym] = entry[1]
        return out


# Singleton
data_fetcher = DataFetcher()
