"""
Bollinger Band Squeeze + Inside Bar Strategy.

Detects volatility contraction (BB squeeze) followed by Inside Bars at or near
the lower Bollinger Band. This combination is a strong bullish reversal setup
that often precedes explosive bounces.

Inside Bar: A candle whose high and low are completely contained within
the previous candle's range.

This is a PROPRIETARY strategy combining two well-known concepts in a
novel way for crypto scalping.
"""
import pandas as pd
from typing import Dict, Optional
from src.indicators.technical import bollinger_bands
from src.strategies.base import BaseStrategy, Signal


class BBSqueezeStrategy(BaseStrategy):
    name = "bb_squeeze_inside_bar"
    weight = 1.3

    def analyze(self, df: pd.DataFrame, symbol: str,
                multi_tf_data: Optional[Dict[str, pd.DataFrame]] = None,
                order_book: Optional[Dict] = None) -> Signal:
        if len(df) < 50:
            return self._neutral("Insufficient data")

        close = df["close"]
        high = df["high"]
        low = df["low"]

        bb = bollinger_bands(close, 20, 2)
        bb_width = bb["bandwidth"]
        bb_pct = bb["percent_b"]
        current_bb_width = bb_width.iloc[-1]
        avg_bb_width = bb_width.iloc[-50:].mean()
        width_ratio = current_bb_width / avg_bb_width if avg_bb_width > 0 else 1

        # Inside bar detection
        if len(df) >= 2:
            prev_high = high.iloc[-2]
            prev_low = low.iloc[-2]
            curr_high = high.iloc[-1]
            curr_low = low.iloc[-1]
            inside_bar = (curr_high <= prev_high and curr_low >= prev_low)
        else:
            inside_bar = False

        current_price = close.iloc[-1]
        bb_lower = bb["lower"].iloc[-1]
        bb_upper = bb["upper"].iloc[-1]
        near_lower = current_price <= bb_lower * 1.01
        bb_pct_val = bb_pct.iloc[-1]

        score = 0
        reasons = []
        details = {
            "bb_width": float(current_bb_width),
            "avg_bb_width": float(avg_bb_width),
            "width_ratio": float(width_ratio),
            "inside_bar": bool(inside_bar),
            "near_lower_band": bool(near_lower),
            "bb_percent_b": float(bb_pct_val),
        }

        # Squeeze: BB width is below 60% of average
        if width_ratio < 0.6:
            score += 30
            reasons.append(f"BB squeeze (width ratio {width_ratio:.2f})")

        # Inside bar
        if inside_bar:
            score += 25
            reasons.append("Inside bar (volatility pause)")

        # Near lower band (oversold)
        if near_lower or bb_pct_val < 0.1:
            score += 30
            reasons.append(f"Price near lower BB (Percent B = {bb_pct_val:.2f})")

        # Both squeeze + inside bar + near lower = strong signal
        if width_ratio < 0.6 and inside_bar and near_lower:
            score += 25
            reasons.append("Triple confluence: Squeeze + Inside Bar + Lower Band")

        if score > 25:
            return self._bull(score, reasons, details)
        return self._neutral(f"No setup (width={width_ratio:.2f}, inside_bar={inside_bar}, near_lower={near_lower})", details)
