"""
TD Sequential Strategy - Tom DeMark's famous 13-bar countdown.

The TD Sequential is one of the most respected indicators in technical analysis.
- Setup phase: 9 consecutive closes below the close 4 bars earlier (for buy setup)
- Countdown phase: 13 closes below the low 2 bars earlier (for buy signal)

A completed countdown (13 bars) is a strong reversal signal.
"""
import pandas as pd
from typing import Dict, Optional
from src.indicators.proprietary import td_sequential
from src.strategies.base import BaseStrategy, Signal


class TDSequentialStrategy(BaseStrategy):
    name = "td_sequential"
    weight = 1.4

    def analyze(self, df: pd.DataFrame, symbol: str,
                multi_tf_data: Optional[Dict[str, pd.DataFrame]] = None,
                order_book: Optional[Dict] = None) -> Signal:
        if len(df) < 30:
            return self._neutral("Insufficient data")

        result = td_sequential(df["close"], setup_length=9, countdown_length=13)
        signal = result.get("signal", "none")
        setup_count = result.get("setup_count", 0)
        countdown_count = result.get("countdown_count", 0)

        details = {
            "setup_count": setup_count,
            "countdown_count": countdown_count,
            "signal": signal,
            "setup_threshold": 9,
            "countdown_threshold": 13,
        }

        if signal == "bullish":
            # Completed countdown = very strong signal
            return self._bull(
                85,
                result.get("reason", "TD Sequential buy countdown complete (13/13)"),
                details
            )
        elif signal == "forming":
            # Partial setup (>= 6/9 or >= 8/13 countdown) = weaker signal
            if countdown_count >= 8:
                score = (countdown_count / 13) * 50
                return self._bull(
                    score,
                    f"TD buy setup complete ({setup_count}/9), countdown {countdown_count}/13",
                    details
                )
            elif setup_count >= 6:
                score = (setup_count / 9) * 30
                return self._bull(
                    score,
                    f"TD buy setup forming ({setup_count}/9)",
                    details
                )
        return self._neutral(f"TD setup: {setup_count}/9, countdown: {countdown_count}/13", details)
