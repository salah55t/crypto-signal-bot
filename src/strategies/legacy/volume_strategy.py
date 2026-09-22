"""
Volume Strategy
Confirms price moves using volume spikes, OBV, MFI, CMF, CVD.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
from src.indicators import volume as vol
from src.strategies.base import BaseStrategy, Signal


class VolumeStrategy(BaseStrategy):
    name = "volume_analysis"
    weight = 1.2

    def analyze(self, df: pd.DataFrame, symbol: str,
                multi_tf_data: Optional[Dict[str, pd.DataFrame]] = None,
                order_book: Optional[Dict] = None) -> Signal:
        if len(df) < 30:
            return self._neutral("Insufficient data")

        # Use volume analyzer
        analysis = vol.analyze_volume(df, period=20)
        if not analysis:
            return self._neutral("Volume analysis failed")

        close = df["close"]
        current_price = close.iloc[-1]
        price_change = (current_price - close.iloc[-2]) / close.iloc[-2] * 100
        score = 0.0
        reasons = []
        details = {
            "current_volume": analysis["current_volume"],
            "avg_volume": analysis["avg_volume"],
            "spike_ratio": analysis["spike_ratio"],
            "is_spike": analysis["is_spike"],
            "cvd": analysis["cvd"],
            "cvd_delta": analysis["cvd_delta"],
            "cvd_trend": analysis["cvd_trend"],
            "volume_trend": analysis["volume_trend"],
            "volume_slope": analysis["volume_slope"],
            "poc": analysis.get("poc"),
        }

        # ============ Volume Spike + Price ============
        if analysis["is_spike"]:
            spike = analysis["spike_ratio"]
            if price_change > 0:
                score += min(30, 15 * spike)
                reasons.append(
                    f"Volume spike {spike:.1f}x with bullish candle (+{price_change:.2f}%)"
                )
            else:
                score -= min(30, 15 * spike)
                reasons.append(
                    f"Volume spike {spike:.1f}x with bearish candle ({price_change:.2f}%)"
                )

        # ============ CVD Trend ============
        if analysis["cvd_trend"] == "up":
            score += 10
            reasons.append("CVD rising (buyers aggressive)")
        elif analysis["cvd_trend"] == "down":
            score -= 10
            reasons.append("CVD falling (sellers aggressive)")

        # ============ Volume Trend ============
        if analysis["volume_trend"] == "rising":
            score += 5
            reasons.append("Volume trend rising")
        elif analysis["volume_trend"] == "falling":
            score -= 5
            reasons.append("Volume trend falling")

        # ============ POC vs Price ============
        poc = analysis.get("poc")
        if poc and current_price < poc * 0.98:
            score += 8
            reasons.append("Price below POC (potential mean reversion up)")
        elif poc and current_price > poc * 1.02:
            score -= 8
            reasons.append("Price above POC (potential mean reversion down)")

        # ============ Divergence: price down + volume down ============
        # Could be exhaustion
        if price_change < -1 and analysis["volume_trend"] == "falling":
            score += 5
            reasons.append("Price down + volume falling (exhaustion selling)")

        score = max(-100, min(100, score))

        if score > 15:
            return self._bull(score, reasons, details)
        elif score < -15:
            return self._bear(abs(score), reasons, details)
        return self._neutral(f"Neutral volume ({score:+.1f})", details)
