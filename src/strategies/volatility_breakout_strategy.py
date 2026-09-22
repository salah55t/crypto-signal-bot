"""
Composite Strategy #3: VOLATILITY BREAKOUT
The Smart Trader's Approach:

"After consolidation comes expansion. Trade the breakout with volume."

A real trader knows that:
  1. Low volatility periods (BB squeeze) are accumulation phases
  2. Breakouts from squeeze with HIGH VOLUME are confirmed moves
  3. ADX must be rising (trend strength emerging)
  4. Direction is determined by which band breaks

This strategy is backtested on BTC over 7.5 years with profit
(Reddit r/algotrading user). The key is CONFIRMATION via volume + ADX.

CONFLUENCE CHECKLIST (need 4/5 to trigger):
  ☐ Bollinger Band Squeeze (width < 60% of 50-period average)
  ☐ Volume buildup before breakout (or spike during breakout)
  ☐ Breakout candle closes outside BB (above upper or below lower)
  ☐ Volume on breakout > 1.5x average (confirmation)
  ☐ ADX > 20 and rising (trend strength emerging)
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
from src.indicators.technical import bollinger_bands, atr, adx, rsi, ema
from src.strategies.base import BaseStrategy, Signal


class VolatilityBreakoutStrategy(BaseStrategy):
    """Composite #3: Volatility Breakout — trade the squeeze breakout with volume confirmation."""
    name = "volatility_breakout"
    weight = 1.8

    def analyze(self, df: pd.DataFrame, symbol: str,
                multi_tf_data: Optional[Dict[str, pd.DataFrame]] = None,
                order_book: Optional[Dict] = None) -> Signal:
        if len(df) < 60:
            return self._neutral("Insufficient data")

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # === Compute indicators ===
        bb = bollinger_bands(close, 20, 2)
        bb_upper = float(bb["upper"].iloc[-1])
        bb_lower = float(bb["lower"].iloc[-1])
        bb_middle = float(bb["middle"].iloc[-1])
        bb_width = float(bb["bandwidth"].iloc[-1])
        avg_bb_width = float(bb["bandwidth"].iloc[-50:].mean())
        squeeze_ratio = bb_width / max(avg_bb_width, 1e-9)

        atr_val = float(atr(high, low, close, 14).iloc[-1])
        adx_df = adx(high, low, close, 14)
        adx_val = float(adx_df["adx"].iloc[-1])
        adx_prev = float(adx_df["adx"].iloc[-2]) if len(adx_df) > 1 else 0
        adx_rising = adx_val > adx_prev

        rsi_val = float(rsi(close, 14).iloc[-1])
        ema_50 = ema(close, 50)
        ema_50_val = float(ema_50.iloc[-1])

        # Volume metrics
        vol_sma = volume.rolling(20).mean()
        vol_now = float(volume.iloc[-1])
        vol_avg = float(vol_sma.iloc[-1])
        vol_ratio = vol_now / max(vol_avg, 1e-9)
        # Volume during last 5 bars (buildup detection)
        recent_vol = float(volume.iloc[-5:].mean())
        recent_vol_ratio = recent_vol / max(vol_avg, 1e-9)

        current_price = float(close.iloc[-1])
        last_open = float(df["open"].iloc[-1])
        last_close = float(close.iloc[-1])
        bullish_candle = last_close > last_open

        # === CONFLUENCE CHECKLIST ===
        score = 0
        reasons = []
        details = {
            "bb_upper": bb_upper, "bb_lower": bb_lower, "bb_middle": bb_middle,
            "bb_width": bb_width, "avg_bb_width": avg_bb_width,
            "squeeze_ratio": float(squeeze_ratio),
            "adx": adx_val, "adx_rising": bool(adx_rising),
            "rsi": rsi_val, "atr": atr_val,
            "volume_ratio": float(vol_ratio),
            "recent_volume_ratio": float(recent_vol_ratio),
        }

        # 1) Bollinger Band Squeeze (width < 60% of average)
        is_squeeze = squeeze_ratio < 0.6
        was_squeezed_recently = False
        # Check if any of last 5 candles was in squeeze
        for i in range(-5, 0):
            if abs(i) <= len(bb["bandwidth"]):
                w = float(bb["bandwidth"].iloc[i])
                if w < avg_bb_width * 0.7:
                    was_squeezed_recently = True
                    break

        if is_squeeze:
            score += 20
            reasons.append(f"Active BB squeeze (ratio {squeeze_ratio:.2f})")
        elif was_squeezed_recently:
            score += 12
            reasons.append(f"Recent BB squeeze (current ratio {squeeze_ratio:.2f})")

        # 2) Volume buildup or spike
        if recent_vol_ratio > 1.3:
            # Volume buildup during squeeze (smart money accumulating)
            score += 15
            reasons.append(f"Volume buildup before breakout ({recent_vol_ratio:.2f}x avg)")
        if vol_ratio > 1.5:
            # Volume spike on breakout candle
            score += 18
            reasons.append(f"Volume spike on breakout ({vol_ratio:.2f}x avg)")

        # 3) Breakout candle (close outside BB)
        bullish_breakout = current_price > bb_upper and bullish_candle
        bearish_breakout = current_price < bb_lower and not bullish_candle

        if bullish_breakout:
            score += 35
            reasons.append(f"BUCKSHOT: close {current_price:.4f} > upper BB {bb_upper:.4f}")
        elif bearish_breakout:
            score += 35
            reasons.append(f"BEARISH breakout: close {current_price:.4f} < lower BB {bb_lower:.4f}")
        elif is_squeeze or was_squeezed_recently:
            # Squeeze but no breakout yet — small bullish bias if above middle
            if current_price > bb_middle:
                score += 8
                reasons.append("Squeeze + price above middle BB (bullish setup)")
            else:
                score += 5
                reasons.append("Squeeze + price below middle BB (bearish setup)")

        # 4) Volume on breakout candle > 1.5x avg (already checked above)
        # Combined into #2

        # 5) ADX > 20 and rising (trend strength emerging)
        if adx_val > 25 and adx_rising:
            score += 18
            reasons.append(f"Strong + rising ADX ({adx_val:.1f}, was {adx_prev:.1f})")
        elif adx_val > 20 and adx_rising:
            score += 12
            reasons.append(f"Rising ADX ({adx_val:.1f} ↑)")
        elif adx_val > 20:
            score += 5
            reasons.append(f"ADX above 20 ({adx_val:.1f})")

        # Bonus: Direction filter (price above EMA 50 for long signals)
        if bullish_breakout:
            if current_price > ema_50_val:
                score += 8
                reasons.append(f"Price above EMA 50 (trend-aligned long)")
            else:
                # Counter-trend breakout, weaker signal
                score -= 5
                reasons.append("Caution: breakout against EMA 50 trend")

        # Bonus: RSI confirms direction
        if bullish_breakout and rsi_val > 50 and rsi_val < 75:
            score += 5
            reasons.append(f"RSI bullish zone ({rsi_val:.1f})")
        elif bearish_breakout and rsi_val < 50 and rsi_val > 25:
            score += 5
            reasons.append(f"RSI bearish zone ({rsi_val:.1f})")

        # Clamp
        score = max(-100, min(100, score))

        # Need at least 50 points (4/5 checklist + bonuses)
        if bullish_breakout and score >= 50:
            return self._bull(score, reasons, details)
        elif bearish_breakout and score >= 50:
            return self._bear(score, reasons, details)
        elif score >= 30 and (is_squeeze or was_squeezed_recently):
            # Squeeze building — small bullish bias
            if current_price > bb_middle:
                return self._bull(score * 0.4, [f"Setup forming: {r}" for r in reasons], details)
        return self._neutral(f"No breakout signal (score={score})", details)
