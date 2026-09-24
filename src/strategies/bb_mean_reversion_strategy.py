"""
v5.7 Composite Strategy #5: BOLLINGER BANDS MEAN REVERSION (user specification)

"تحقيق أرباح سريعة عند انحراف السعر عن متوسّطه الطبيعي في الأسواق ذات
التقلب أو التجميع"

Signal Stack Framework (the golden rule - one indicator per class):
  VOLATILITY     : Bollinger Bands (20, 2)  (where is the statistical edge?)
  MOMENTUM       : Stochastic (14, 3, 3)    (is it oversold + turning?)
  CONFIRMATION   : bullish candle + volume  (real buyers, not a dead cat)

Automatic BUY conditions (all from the user spec):
  1. Price touches or breaks the LOWER Bollinger Band
  2. Stochastic %K < 20 (oversold) — RSI < 30 accepted as the alternative
     reading, both together score higher
  3. FRESH positive %K x %D cross INSIDE the oversold zone
  Bonus: last candle closed bullish + volume present (capitulation/bounce)

Automatic SELL/EXIT conditions (user spec, emitted as bearish signals):
  - Price reaches the UPPER band while RSI > 70  -> mean reversion complete,
    close the trade (contributes to opposite-signal exits)
  - TP ladder note: the bot's own TP1 (banks 50%) plays the role of the
    MIDDLE band target and TP2 the upper-band full exit; the exact BB levels
    are carried in `details` for transparency.
"""
import pandas as pd
from typing import Dict, Optional
from src.indicators.technical import (
    bollinger_bands, stochastic, rsi, sma, atr,
)
from src.strategies.base import BaseStrategy, Signal

# Slow stochastic (14, 3, 3): raw %K smoothed 3, then %D = SMA(slow %K, 3)
K_PERIOD, SMOOTH, D_PERIOD = 14, 3, 3


class BBMeanReversionStrategy(BaseStrategy):
    """Composite #5: fade the band extremes back to the mean."""
    name = "bb_mean_reversion"

    def analyze(self, df: pd.DataFrame, symbol: str,
                multi_tf_data: Optional[Dict[str, pd.DataFrame]] = None,
                order_book: Optional[Dict] = None) -> Signal:
        if len(df) < 60:
            return self._neutral("Insufficient data")

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # === Signal Stack: one indicator per class ===
        bb = bollinger_bands(close, 20, 2)
        raw_k = stochastic(high, low, close, K_PERIOD, D_PERIOD)["k"]
        slow_k = sma(raw_k, SMOOTH)          # (14, 3, 3) smoothing
        slow_d = sma(slow_k, D_PERIOD)
        rsi_series = rsi(close, 14)
        atr_val = float(atr(high, low, close, 14).iloc[-1])

        price = float(close.iloc[-1])
        lower = float(bb["lower"].iloc[-1])
        upper = float(bb["upper"].iloc[-1])
        middle = float(bb["middle"].iloc[-1])
        pct_b = float(bb["percent_b"].iloc[-1])
        k_now = float(slow_k.iloc[-1])
        d_now = float(slow_d.iloc[-1])
        k_prev = float(slow_k.iloc[-2])
        d_prev = float(slow_d.iloc[-2])
        rsi_now = float(rsi_series.iloc[-1])
        vol_ratio = float(volume.iloc[-1] / max(
            float(volume.rolling(20).mean().iloc[-1]), 1e-9))

        details = {
            "stack": {"volatility": "BB(20,2)", "momentum": "Stoch(14,3,3)",
                      "confirmation": "candle+volume"},
            "price": price, "bb_lower": lower, "bb_middle": middle,
            "bb_upper": upper, "percent_b": pct_b,
            "stoch_k": k_now, "stoch_d": d_now, "rsi": rsi_now,
            "volume_ratio": vol_ratio, "atr": atr_val,
            "tp1_middle_band": middle, "tp2_upper_band": upper,
        }

        # === EXIT SIDE (user spec: upper band + RSI > 70 = full exit) ===
        if price >= upper and rsi_now > 70:
            return self._bear(
                55, f"Upper BB reached + RSI overbought ({rsi_now:.1f}) - "
                    f"mean reversion complete", details)

        # === BUY CHECKLIST (Signal Stack) ===
        score = 0
        reasons = []

        # --- VOLATILITY class (mandatory gate) ---
        # 1) price touched or broke the lower band
        band_touch = float(low.iloc[-1]) <= lower or pct_b < 0.05
        band_break = price < lower
        if not band_touch:
            return self._neutral(
                f"Price not at lower BB (percent B = {pct_b:.2f})", details)
        score += 30 if band_break else 25
        reasons.append(
            f"Lower BB {'broken' if band_break else 'touched'} "
            f"(percent B = {pct_b:.2f})")

        # --- MOMENTUM class ---
        # 2) oversold reading
        stoch_oversold = k_now < 20
        rsi_oversold = rsi_now < 30
        if stoch_oversold and rsi_oversold:
            score += 25
            reasons.append(f"Deeply oversold (Stoch {k_now:.1f}, RSI {rsi_now:.1f})")
        elif stoch_oversold:
            score += 20
            reasons.append(f"Stochastic oversold ({k_now:.1f} < 20)")
        elif rsi_oversold:
            score += 15
            reasons.append(f"RSI oversold ({rsi_now:.1f} < 30)")
        else:
            return self._neutral(
                f"No oversold reading (Stoch {k_now:.1f}, RSI {rsi_now:.1f})",
                details)

        # 3) fresh positive %K x %D cross inside the oversold zone
        fresh_cross = (k_prev <= d_prev) and (k_now > d_now)
        if fresh_cross and k_now < 30:
            score += 30
            reasons.append(
                f"Fresh Stoch cross up in oversold zone "
                f"({k_prev:.1f}/{d_prev:.1f} -> {k_now:.1f}/{d_now:.1f})")
        elif fresh_cross:
            score += 15
            reasons.append("Stoch cross up (outside deep oversold)")
        elif k_now > d_now and k_now < 25:
            score += 10
            reasons.append("Stoch turning up in oversold zone")
        else:
            return self._neutral("No Stoch bullish crossover yet", details)

        # --- CONFIRMATION class ---
        # bullish candle + real volume (not a dead-cat bounce)
        if float(close.iloc[-1]) > float(df["open"].iloc[-1]):
            score += 10
            reasons.append("Bullish close (no falling knife)")
        if vol_ratio >= 1.5:
            score += 10
            reasons.append(f"Capitulation volume ({vol_ratio:.2f}x avg20)")
        elif vol_ratio >= 0.8:
            score += 5
            reasons.append(f"Volume present ({vol_ratio:.2f}x)")

        score = max(-100, min(100, score))

        if score >= 60:
            return self._bull(score, reasons, details)
        elif score >= 40:
            return self._bull(score * 0.5,
                              [f"Partial mean-reversion: {r}" for r in reasons],
                              details)
        return self._neutral(f"Score too low ({score})", details)
