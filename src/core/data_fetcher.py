"""Data fetcher - retrieves and formats market data from Binance."""
import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from src.core.binance_client import binance_client
from src.utils.logger import log
from src.utils.helpers import retry_on_failure


class DataFetcher:
    """Fetches OHLCV data and order book snapshots for analysis."""

    INTERVALS = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h",
                 "6h", "8h", "12h", "1d", "3d", "1w", "1M"}

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
        """Get historical candles for a symbol as a DataFrame."""
        if interval not in DataFetcher.INTERVALS:
            raise ValueError(f"Invalid interval '{interval}'. Valid: {DataFetcher.INTERVALS}")
        raw = binance_client.get_klines(symbol, interval, limit=limit)
        return DataFetcher.klines_to_df(raw)

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
        """Get order book with computed metrics."""
        ob = binance_client.get_order_book(symbol, limit)
        bids = pd.DataFrame(ob.get("bids", []), columns=["price", "qty"], dtype=float)
        asks = pd.DataFrame(ob.get("asks", []), columns=["price", "qty"], dtype=float)
        return {
            "symbol": symbol,
            "bids": bids,
            "asks": asks,
            "last_update_id": ob.get("lastUpdateId"),
        }

    @staticmethod
    @retry_on_failure
    def get_ticker_24h(symbol: str) -> Dict:
        """Get 24h ticker statistics."""
        return binance_client.get_ticker(symbol)

    @staticmethod
    @retry_on_failure
    def get_batch_tickers(symbols: List[str]) -> Dict[str, Dict]:
        """Get 24h tickers for many symbols in one call."""
        all_t = binance_client.get_all_tickers()
        sym_set = set(symbols)
        return {t["symbol"]: t for t in all_t if t["symbol"] in sym_set}


# Singleton
data_fetcher = DataFetcher()
