"""
Wyckoff Spring Strategy - Detects institutional accumulation pattern.
A Wyckoff Spring occurs when price briefly breaks below a support level
(spring) and then quickly recovers above it. This indicates that "smart money"
has absorbed the selling pressure at lower prices.

This is a PROPRIETARY strategy inspired by Richard Wyckoff's methodology,
adapted for crypto markets with tighter tolerances.
"""
import pandas as pd
from typing import Dict, Optional
from src.indicators.proprietary import wyckoff_spring
from src.strategies.base import BaseStrategy, Signal


class WyckoffSpringStrategy(BaseStrategy):
    name = "wyckoff_spring"
    weight = 1.6  # heavy weight - strong bottom signal

    def analyze(self, df: pd.DataFrame, symbol: str,
                multi_tf_data: Optional[Dict[str, pd.DataFrame]] = None,
                order_book: Optional[Dict] = None) -> Signal:
        if len(df) < 60:
            return self._neutral("Insufficient data")

        result = wyckoff_spring(df, support_lookback=50,
                                  breakdown_threshold=0.005,
                                  recovery_threshold=0.5)
        signal = result.get("signal", "none")
        strength = result.get("strength", 0)

        details = {
            "support_level": result.get("support"),
            "breakdown_low": result.get("breakdown_low"),
            "recovery_pct": result.get("recovery_pct"),
        }

        if signal == "bullish":
            score = strength * 80  # up to 80 points
            return self._bull(
                score,
                result.get("reason", "Wyckoff Spring detected"),
                details
            )
        return self._neutral(f"No spring (support at {result.get('support')})", details)
