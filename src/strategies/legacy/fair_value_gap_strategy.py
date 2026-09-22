"""
Fair Value Gap (FVG) Strategy - Detects 3-candle imbalance patterns.

FVGs are price gaps that haven't been filled. They act as magnets — price
tends to revisit them. Unfilled bullish FVGs above current price act as
support when revisited.

Based on ICT methodology — high-probability reversal zones.
"""
import pandas as pd
from typing import Dict, Optional
from src.indicators.ict import detect_active_fvg
from src.strategies.base import BaseStrategy, Signal


class FairValueGapStrategy(BaseStrategy):
    name = "fair_value_gap"
    weight = 1.6

    def analyze(self, df: pd.DataFrame, symbol: str,
                multi_tf_data: Optional[Dict[str, pd.DataFrame]] = None,
                order_book: Optional[Dict] = None) -> Signal:
        if len(df) < 60:
            return self._neutral("Insufficient data")

        result = detect_active_fvg(df, lookback=50)
        signal = result.get("signal", "none")
        strength = result.get("strength", 0)

        if signal == "bullish":
            score = strength * 70
            return self._bull(score, result.get("reason", ""), result.get("details", {}))
        elif signal == "bearish":
            score = strength * 70
            return self._bear(score, result.get("reason", ""), result.get("details", {}))
        return self._neutral("No active FVG", result.get("details", {}))
