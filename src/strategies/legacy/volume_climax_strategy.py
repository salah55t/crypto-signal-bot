"""
Volume Climax & Exhaustion Strategy - Detects selling climax and capitulation.

A Volume Climax is an extremely high volume spike combined with a sharp price
drop, often signaling that sellers have exhausted themselves. This is a
classic "capitulation" pattern that marks market bottoms.

Combined with Volatility Contraction (ATR ratio shrinking), it identifies
periods where sellers are exhausted and a reversal is imminent.
"""
import pandas as pd
from typing import Dict, Optional
from src.indicators.proprietary import volume_climax, volatility_contraction
from src.strategies.base import BaseStrategy, Signal


class VolumeClimaxStrategy(BaseStrategy):
    name = "volume_climax_exhaustion"
    weight = 1.5

    def analyze(self, df: pd.DataFrame, symbol: str,
                multi_tf_data: Optional[Dict[str, pd.DataFrame]] = None,
                order_book: Optional[Dict] = None) -> Signal:
        if len(df) < 60:
            return self._neutral("Insufficient data")

        climax = volume_climax(df, lookback=50, spike_threshold=2.5,
                                drop_threshold=-2.0)
        contraction = volatility_contraction(df, lookback=20,
                                              contraction_ratio=0.6)

        score = 0
        reasons = []
        details = {"climax": climax, "contraction": contraction}

        if climax.get("signal") == "bullish":
            score = climax.get("strength", 0.5) * 70
            reasons.append(climax.get("reason", ""))
        if contraction.get("signal") == "bullish":
            score += contraction.get("strength", 0.5) * 50
            reasons.append(contraction.get("reason", ""))

        if score > 20:
            return self._bull(score, reasons, details)
        return self._neutral("No climax/contraction detected", details)
