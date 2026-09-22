"""
Smart Money Divergence Strategy - Detects accumulation by institutions.

When Volume-Weighted RSI is significantly higher than regular RSI at low
prices, it indicates that "smart money" is buying despite the bearish price
action. This is a powerful early indicator of upcoming reversals.

PROPRIETARY: Combines volume-weighted RSI calculation with divergence logic.
"""
import pandas as pd
from typing import Dict, Optional
from src.indicators.proprietary import smart_money_divergence, volume_weighted_rsi
from src.indicators.technical import rsi
from src.strategies.base import BaseStrategy, Signal


class SmartMoneyStrategy(BaseStrategy):
    name = "smart_money_divergence"
    weight = 1.4

    def analyze(self, df: pd.DataFrame, symbol: str,
                multi_tf_data: Optional[Dict[str, pd.DataFrame]] = None,
                order_book: Optional[Dict] = None) -> Signal:
        if len(df) < 50:
            return self._neutral("Insufficient data")

        close = df["close"]
        volume = df["volume"]
        if volume.sum() == 0:
            return self._neutral("No volume data")

        result = smart_money_divergence(close, volume, lookback=50)
        signal = result.get("signal", "none")
        strength = result.get("strength", 0)

        # Additional confirmation: regular RSI should be oversold
        rsi_val = rsi(close, 14).iloc[-1]
        details = result.get("details", {})
        details["current_rsi"] = float(rsi_val)

        if signal == "bullish":
            # Stronger signal when RSI is also oversold
            if rsi_val < 35:
                score = strength * 80
                reasons = [result.get("reason", ""), f"RSI oversold ({rsi_val:.1f})"]
            else:
                score = strength * 50
                reasons = [result.get("reason", "")]
            return self._bull(score, reasons, details)
        return self._neutral(f"No smart money divergence (RSI={rsi_val:.1f})", details)
