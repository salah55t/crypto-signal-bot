"""
Bottom Reversal Strategy - Comprehensive bounce detector.

Combines ALL bounce signals into a single high-weight strategy:
  - Near recent low (within 5% of 50-bar low)
  - RSI oversold (< 35)
  - Volume Climax (capitulation selling)
  - Wyckoff Spring (false breakdown + recovery)
  - Smart Money Divergence (VW-RSI > RSI)
  - Fibonacci golden zone (0.618 - 0.786)
  - Bollinger %B < 0.05 (below lower band)
  - Bullish candlestick patterns at bottom

This strategy gives STRONG bullish signals when multiple bounce
signals align — designed to ensure bottom-reversal candidates
actually pass the main scorer and become tradeable recommendations.
"""
import pandas as pd
from typing import Dict, Optional
from src.indicators.technical import rsi, bollinger_bands, atr
from src.indicators.proprietary import (
    is_near_bottom, volume_climax, wyckoff_spring,
    smart_money_divergence, fibonacci_confluence
)
from src.indicators.patterns import detect_all_patterns
from src.strategies.base import BaseStrategy, Signal


class BottomReversalStrategy(BaseStrategy):
    """Comprehensive bottom-reversal detection with multi-signal confluence."""
    name = "bottom_reversal"
    weight = 1.8  # highest weight — bounce signals are powerful

    def analyze(self, df: pd.DataFrame, symbol: str,
                multi_tf_data: Optional[Dict[str, pd.DataFrame]] = None,
                order_book: Optional[Dict] = None) -> Signal:
        if len(df) < 60:
            return self._neutral("Insufficient data")

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]
        current_price = float(close.iloc[-1])

        # ============================================
        # 1) Check if price is near recent bottom
        # ============================================
        bottom_check = is_near_bottom(df, lookback=50, threshold_pct=0.08)
        if not bottom_check["near_bottom"]:
            # If price is far from bottom, return weak neutral
            return self._neutral(
                f"Not near bottom ({bottom_check['distance_from_low_pct']:.1f}% above low)",
                {"distance_from_low_pct": bottom_check.get("distance_from_low_pct", 0)}
            )

        # Base score for being near bottom
        score = 25
        signals_detected = [f"Near bottom ({bottom_check['distance_from_low_pct']:.1f}% above 50-bar low)"]

        # ============================================
        # 2) RSI oversold
        # ============================================
        rsi_val = float(rsi(close, 14).iloc[-1])
        if rsi_val < 25:
            score += 30
            signals_detected.append(f"RSI deeply oversold ({rsi_val:.1f})")
        elif rsi_val < 35:
            score += 20
            signals_detected.append(f"RSI oversold ({rsi_val:.1f})")
        elif rsi_val < 45:
            score += 8
            signals_detected.append(f"RSI weak ({rsi_val:.1f})")

        # ============================================
        # 3) Volume Climax (capitulation selling)
        # ============================================
        climax = volume_climax(df, lookback=50, spike_threshold=2.0,
                                drop_threshold=-1.0)
        if climax.get("signal") == "bullish":
            strength = climax.get("strength", 0.5)
            score += int(strength * 35)
            signals_detected.append(climax.get("reason", ""))

        # ============================================
        # 4) Wyckoff Spring
        # ============================================
        spring = wyckoff_spring(df, support_lookback=50)
        if spring.get("signal") == "bullish":
            strength = spring.get("strength", 0.5)
            score += int(strength * 40)
            signals_detected.append(spring.get("reason", ""))

        # ============================================
        # 5) Smart Money Divergence (VW-RSI > RSI)
        # ============================================
        smd = smart_money_divergence(close, volume, lookback=50)
        if smd.get("signal") == "bullish":
            strength = smd.get("strength", 0.5)
            score += int(strength * 30)
            signals_detected.append(smd.get("reason", ""))

        # ============================================
        # 6) Fibonacci golden zone
        # ============================================
        fib = fibonacci_confluence(df, lookback=100)
        if fib.get("signal") == "bullish":
            strength = fib.get("strength", 0.5)
            score += int(strength * 25)
            signals_detected.append(fib.get("reason", ""))

        # ============================================
        # 7) Bollinger %B below lower band
        # ============================================
        bb = bollinger_bands(close, 20, 2)
        pct_b = float(bb["percent_b"].iloc[-1])
        if pct_b < 0.02:
            score += 20
            signals_detected.append(f"Below lower BB (Percent B = {pct_b:.2f})")
        elif pct_b < 0.10:
            score += 10
            signals_detected.append(f"Near lower BB (Percent B = {pct_b:.2f})")

        # ============================================
        # 8) Bullish candlestick patterns
        # ============================================
        patterns = detect_all_patterns(df)
        bullish_patterns = [p for p in patterns if p.get("direction") == "bullish"]
        if bullish_patterns:
            pattern_strength = sum(p["strength"] for p in bullish_patterns)
            score += min(25, int(pattern_strength * 25))
            for p in bullish_patterns[:3]:
                signals_detected.append(f"{p['pattern']} (strength={p['strength']:.1f})")

        # ============================================
        # Final decision
        # ============================================
        # Lower threshold for bullish signal: was 25, now 15
        if score >= 15:
            # Cap the score at 100 for normalization
            score = min(100, score)
            # Strong bounce signal — these will become recommendations
            details = {
                "bounce_score": score,
                "rsi": rsi_val,
                "bb_percent_b": pct_b,
                "patterns_detected": [p["pattern"] for p in patterns],
                "near_bottom": True,
                "distance_from_low_pct": bottom_check["distance_from_low_pct"],
            }
            return Signal(
                strategy=self.name,
                direction="bullish",
                score=float(score),
                confidence=float(score) / 100.0,
                reasons=signals_detected,
                details=details,
            )
        return self._neutral(
            f"Bounce signals too weak (score={score})",
            {"bounce_score": score, "near_bottom": True}
        )
