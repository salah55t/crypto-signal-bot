"""
ICT Confluence Strategy - Combines OB + FVG + Liquidity Sweep.

When 2+ of these signals align in the same direction, the probability of
a successful trade increases exponentially. This is the highest-probability
setup in ICT methodology.

Per ICT traders:
"The Confirmation Model: OB + FVG + Liquidity Sweep - When an Order Block
sits directly adjacent to or inside an active Fair Value Gap, the probability
of a successful trade increases exponentially."
"""
import pandas as pd
from typing import Dict, Optional
from src.indicators.ict import detect_ict_confluence
from src.strategies.base import BaseStrategy, Signal


class ICTConfluenceStrategy(BaseStrategy):
    name = "ict_confluence"
    weight = 2.0  # HIGHEST weight — when all 3 align, very high probability

    def analyze(self, df: pd.DataFrame, symbol: str,
                multi_tf_data: Optional[Dict[str, pd.DataFrame]] = None,
                order_book: Optional[Dict] = None) -> Signal:
        if len(df) < 60:
            return self._neutral("Insufficient data")

        result = detect_ict_confluence(df, lookback=50)
        signal = result.get("signal", "none")
        strength = result.get("strength", 0)

        if signal == "bullish":
            # Multi-signal confluence = highest probability
            score = strength * 90  # max ~85 points
            return self._bull(score, result.get("reason", ""), result.get("details", {}))
        elif signal == "bearish":
            score = strength * 90
            return self._bear(score, result.get("reason", ""), result.get("details", {}))
        return self._neutral("No ICT confluence", result.get("details", {}))
