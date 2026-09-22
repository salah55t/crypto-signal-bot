"""
Composite Strategy #1: TREND PULLBACK
The Smart Trader's Approach:

"Buy strength, not weakness. Wait for pullbacks in strong uptrends."

A real trader doesn't buy the bottom (impossible to time). They:
  1. Identify a STRONG trend (multi-EMA alignment + high ADX)
  2. Wait for a pullback to support (EMA 21, the working average)
  3. Confirm with a bullish reversal candle (Hammer/Engulfing)
  4. Verify volume is healthy (not panic selling)
  5. Enter with tight SL below the pullback low

This is the highest-probability trade in trend following.

CONFLUENCE CHECKLIST (need 4/5 to trigger):
  ☐ Strong uptrend: EMA 9 > 21 > 50 + price above EMA 50
  ☐ ADX > 25 (trend strength confirmed)
  ☐ Pullback: price dipped to/near EMA 21 then recovered
  ☐ Bullish reversal candle (Hammer, Bullish Engulfing, Piercing)
  ☐ Healthy volume (not climax, not too low)
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
from src.indicators.technical import ema, rsi, macd, adx, atr, bollinger_bands
from src.indicators.patterns import detect_hammer, detect_engulfing, detect_piercing_line, detect_morning_star
from src.strategies.base import BaseStrategy, Signal


class TrendPullbackStrategy(BaseStrategy):
    """Composite #1: Trend Pullback — buy strength on pullbacks."""
    name = "trend_pullback"
    weight = 2.0  # highest weight

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
        ema_9 = ema(close, 9)
        ema_21 = ema(close, 21)
        ema_50 = ema(close, 50)
        adx_df = adx(high, low, close, 14)
        rsi_val = float(rsi(close, 14).iloc[-1])
        macd_df = macd(close, 12, 26, 9)
        macd_hist = float(macd_df["histogram"].iloc[-1])
        macd_hist_prev = float(macd_df["histogram"].iloc[-2]) if len(macd_df) > 1 else 0
        atr_val = float(atr(high, low, close, 14).iloc[-1])
        vol_sma = volume.rolling(20).mean()
        vol_ratio = float(volume.iloc[-1] / max(vol_sma.iloc[-1], 1e-9))

        # Current values
        current_price = float(close.iloc[-1])
        e9 = float(ema_9.iloc[-1])
        e21 = float(ema_21.iloc[-1])
        e50 = float(ema_50.iloc[-1])
        adx_val = float(adx_df["adx"].iloc[-1])
        plus_di = float(adx_df["plus_di"].iloc[-1])
        minus_di = float(adx_df["minus_di"].iloc[-1])

        # === CONFLUENCE CHECKLIST ===
        score = 0
        reasons = []
        details = {
            "ema_9": e9, "ema_21": e21, "ema_50": e50,
            "adx": adx_val, "plus_di": plus_di, "minus_di": minus_di,
            "rsi": rsi_val, "macd_hist": macd_hist,
            "volume_ratio": vol_ratio, "atr": atr_val,
            "current_price": current_price,
        }

        # 1) Strong uptrend: EMA 9 > 21 > 50 + price above EMA 50
        if e9 > e21 > e50 and current_price > e50:
            score += 25
            reasons.append(f"Strong uptrend (EMA 9 > 21 > 50, price above EMA 50)")
        elif e9 > e21 and current_price > e21:
            score += 12  # partial
            reasons.append(f"Uptrend (EMA 9 > 21)")
        else:
            return self._neutral("No uptrend", details)

        # 2) ADX > 25 (trend strength)
        if adx_val > 30:
            score += 20
            reasons.append(f"Very strong trend (ADX={adx_val:.1f})")
        elif adx_val > 25:
            score += 15
            reasons.append(f"Strong trend (ADX={adx_val:.1f})")
        elif adx_val > 20:
            score += 5
            reasons.append(f"Moderate trend (ADX={adx_val:.1f})")
        else:
            return self._neutral(f"Weak trend (ADX={adx_val:.1f})", details)

        # 3) Pullback to EMA 21 (price dipped below EMA 9 then recovered)
        # Pullback = low of last 3 candles touched/near EMA 21
        recent_low = float(low.iloc[-3:].min())
        pullback_to_ema21 = recent_low <= e21 * 1.005 and recent_low >= e21 * 0.99
        # Price recovered above EMA 9
        recovered = current_price > e9 * 0.999

        if pullback_to_ema21 and recovered:
            score += 25
            reasons.append(f"Pullback to EMA 21 ({e21:.4f}) + recovered")
        elif recent_low < e9 and recovered:
            score += 12
            reasons.append(f"Pullback to EMA 9 ({e9:.4f}) + recovered")

        # 4) Bullish reversal candle (Hammer, Engulfing, Piercing, Morning Star)
        last = df.iloc[-1]
        prev = df.iloc[-2]
        prev2 = df.iloc[-3]
        bullish_candle = False
        pattern_name = ""

        if detect_hammer(last["open"], last["high"], last["low"], last["close"]):
            bullish_candle = True
            pattern_name = "Hammer"
            score += 20
        elif detect_engulfing(prev["open"], prev["high"], prev["low"], prev["close"],
                              last["open"], last["high"], last["low"], last["close"]) == "bullish":
            bullish_candle = True
            pattern_name = "Bullish Engulfing"
            score += 22
        elif detect_piercing_line(prev["open"], prev["high"], prev["low"], prev["close"],
                                    last["open"], last["high"], last["low"], last["close"]):
            bullish_candle = True
            pattern_name = "Piercing Line"
            score += 18
        elif detect_morning_star(prev2["open"], prev2["high"], prev2["low"], prev2["close"],
                                   prev["open"], prev["high"], prev["low"], prev["close"],
                                   last["open"], last["high"], last["low"], last["close"]):
            bullish_candle = True
            pattern_name = "Morning Star"
            score += 25

        if bullish_candle:
            reasons.append(f"Bullish reversal candle: {pattern_name}")

        # 5) Healthy volume (not panic, not dead)
        if 0.7 <= vol_ratio <= 2.0:
            score += 10
            reasons.append(f"Healthy volume (ratio {vol_ratio:.2f}x avg)")
        elif vol_ratio > 2.0:
            # High volume on bullish candle = strong buying
            if last["close"] > last["open"]:
                score += 12
                reasons.append(f"High volume bullish candle ({vol_ratio:.2f}x)")
            else:
                reasons.append(f"High volume but bearish candle")
        else:
            reasons.append(f"Low volume ({vol_ratio:.2f}x)")

        # Bonus: RSI in healthy zone (40-60)
        if 40 <= rsi_val <= 60:
            score += 5
            reasons.append(f"RSI healthy ({rsi_val:.1f})")

        # MACD bullish cross or rising histogram
        if macd_hist > 0 and macd_hist > macd_hist_prev:
            score += 8
            reasons.append("MACD histogram rising")

        # Clamp
        score = max(-100, min(100, score))

        # Need at least 50 points to trigger (4/5 checklist)
        if score >= 50:
            return self._bull(score, reasons, details)
        elif score >= 30:
            return self._bull(score * 0.5, [f"Partial signal: {r}" for r in reasons], details)
        return self._neutral(f"Score too low ({score})", details)
