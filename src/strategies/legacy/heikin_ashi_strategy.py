"""
Heikin Ashi Smoke Strategy - Detects small-body HA candles after downtrend.

The "Smoke" pattern occurs when Heikin Ashi candles show very small bodies
(with long lower wicks) after a sustained downtrend. This is a powerful early
reversal signal that often precedes major bounces.

This is a PROPRIETARY strategy, not commonly found in standard libraries.
"""
import pandas as pd
from typing import Dict, Optional
from src.indicators.proprietary import heikin_ashi, ha_smoke_signal
from src.strategies.base import BaseStrategy, Signal


class HeikinAshiSmokeStrategy(BaseStrategy):
    name = "heikin_ashi_smoke"
    weight = 1.2

    def analyze(self, df: pd.DataFrame, symbol: str,
                multi_tf_data: Optional[Dict[str, pd.DataFrame]] = None,
                order_book: Optional[Dict] = None) -> Signal:
        if len(df) < 30:
            return self._neutral("Insufficient data")

        ha = heikin_ashi(df)
        result = ha_smoke_signal(ha, lookback=10)

        signal = result.get("signal", "none")
        strength = result.get("strength", 0)
        details = {
            "last_ha_open": float(ha["open"].iloc[-1]),
            "last_ha_close": float(ha["close"].iloc[-1]),
            "last_ha_high": float(ha["high"].iloc[-1]),
            "last_ha_low": float(ha["low"].iloc[-1]),
        }

        if signal == "bullish":
            score = strength * 70
            return self._bull(score, result.get("reason", ""), details)
        return self._neutral("No HA smoke signal", details)
