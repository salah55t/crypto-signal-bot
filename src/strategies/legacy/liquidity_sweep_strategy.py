"""
Liquidity Sweep Strategy - Detects stop hunts at key levels.

A Liquidity Sweep occurs when price briefly takes out a previous swing high/low
(triggering stop losses) and then immediately reverses. This is one of the
highest-probability reversal patterns in trading.

Institutions use this to:
  1. Fill large orders at better prices
  2. Stop out retail traders
  3. Reverse the market direction

Based on ICT methodology, used by professional traders.
"""
import pandas as pd
from typing import Dict, Optional
from src.indicators.ict import detect_liquidity_sweep
from src.strategies.base import BaseStrategy, Signal


class LiquiditySweepStrategy(BaseStrategy):
    name = "liquidity_sweep"
    weight = 1.9  # highest weight - one of the most profitable patterns

    def analyze(self, df: pd.DataFrame, symbol: str,
                multi_tf_data: Optional[Dict[str, pd.DataFrame]] = None,
                order_book: Optional[Dict] = None) -> Signal:
        if len(df) < 40:
            return self._neutral("Insufficient data")

        result = detect_liquidity_sweep(df, lookback=30)
        signal = result.get("signal", "none")
        strength = result.get("strength", 0)

        if signal == "bullish":
            # Strong signal: stop hunt + reversal = high probability bounce
            score = strength * 85
            return self._bull(score, result.get("reason", ""), result.get("details", {}))
        elif signal == "bearish":
            score = strength * 85
            return self._bear(score, result.get("reason", ""), result.get("details", {}))
        return self._neutral("No liquidity sweep", result.get("details", {}))
