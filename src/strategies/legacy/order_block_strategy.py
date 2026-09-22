"""
Order Block Strategy - Detects institutional footprints.

Order Blocks are the last opposite-colored candle before a strong directional
move. They represent where institutions placed large orders, and price tends
to revisit these levels.

PROPRIETARY: Based on ICT (Inner Circle Trader) methodology, proven by
professional traders. Detects bullish/bearish OBs and checks if price is
currently testing one.
"""
import pandas as pd
from typing import Dict, Optional
from src.indicators.ict import detect_active_order_block
from src.strategies.base import BaseStrategy, Signal


class OrderBlockStrategy(BaseStrategy):
    name = "order_block"
    weight = 1.7  # high weight - institutional footprint

    def analyze(self, df: pd.DataFrame, symbol: str,
                multi_tf_data: Optional[Dict[str, pd.DataFrame]] = None,
                order_book: Optional[Dict] = None) -> Signal:
        if len(df) < 60:
            return self._neutral("Insufficient data")

        result = detect_active_order_block(df, lookback=50)
        signal = result.get("signal", "none")
        strength = result.get("strength", 0)

        if signal == "bullish":
            score = strength * 75
            return self._bull(score, result.get("reason", ""), result.get("details", {}))
        elif signal == "bearish":
            score = strength * 75
            return self._bear(score, result.get("reason", ""), result.get("details", {}))
        return self._neutral("No active order block", result.get("details", {}))
