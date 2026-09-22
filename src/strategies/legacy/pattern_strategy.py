"""
Candlestick Pattern Strategy
Detects bullish/bearish reversal and continuation patterns.
"""
import pandas as pd
from typing import Dict, Optional
from src.indicators import patterns as pat
from src.strategies.base import BaseStrategy, Signal


class PatternStrategy(BaseStrategy):
    name = "candlestick_patterns"
    weight = 1.0

    def analyze(self, df: pd.DataFrame, symbol: str,
                multi_tf_data: Optional[Dict[str, pd.DataFrame]] = None,
                order_book: Optional[Dict] = None) -> Signal:
        if len(df) < 3:
            return self._neutral("Insufficient data")

        detected = pat.detect_all_patterns(df, lookback=3)
        if not detected:
            return self._neutral("No candlestick patterns detected")

        # Aggregate signals
        bullish_patterns = [p for p in detected if p["direction"] == "bullish"]
        bearish_patterns = [p for p in detected if p["direction"] == "bearish"]
        neutral_patterns = [p for p in detected if p["direction"] == "neutral"]

        bull_score = sum(p["strength"] for p in bullish_patterns) * 50
        bear_score = sum(p["strength"] for p in bearish_patterns) * 50
        net_score = bull_score - bear_score

        reasons = []
        for p in detected:
            reasons.append(
                f"{p['pattern']} ({p['direction']}, strength={p['strength']:.1f})"
            )

        details = {
            "patterns": detected,
            "bullish_count": len(bullish_patterns),
            "bearish_count": len(bearish_patterns),
            "neutral_count": len(neutral_patterns),
        }

        # Lowered thresholds: was >20, now >8 (much less strict)
        if net_score > 8:
            return self._bull(net_score, reasons, details)
        elif net_score < -8:
            return self._bear(abs(net_score), reasons, details)
        return self._neutral(f"Pattern signals mixed ({net_score:+.1f})", details)
