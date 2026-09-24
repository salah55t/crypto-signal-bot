"""
v5.7 Composite Strategy #4: TRIPLE CONFLUENCE TREND (user specification)

"ركوب الموجات الصاعدة الممتدة وتقليل الإشارات الخاطئة أثناء الحركة العرضية"

Signal Stack Framework (the golden rule - one indicator per class):
  DIRECTION      : EMA 200 + EMA 50   (where is the market going?)
  MOMENTUM       : RSI 14             (timing + strength of the entry)
  LIQUIDITY      : Volume vs 20-avg + OBV slope (real traders backing it?)

Automatic BUY conditions (all from the user spec):
  1. Price trades ABOVE EMA 200            (macro uptrend filter)
  2. EMA 50 above EMA 200                  (golden-cross state; a FRESH
     cross within the last 5 bars scores higher)
  3. RSI crosses ABOVE 50                  (fresh momentum ignition), and
     RSI must NOT be above 70              (never buy into overbought)
  4. Volume above its 20-bar average       (liquidity confirmation)
  Bonus: OBV rising over the last 5 bars   (accumulation)

Automatic SELL/EXIT conditions (emitted as bearish signals so the v5.5
opposite-signal machinery can act on them - immediate close >= 55 conf):
  1. Price breaks BELOW EMA 50 while EMA 50 is still above EMA 200
     (trend wave broken - exit the ride)
  2. RSI > 70 IN PARALLEL with a bearish divergence
     (price higher-high, RSI lower-high = exhaustion)
"""
import pandas as pd
from typing import Dict, Optional
from src.indicators.technical import (
    ema, rsi, obv, volume_sma, detect_divergence,
)
from src.strategies.base import BaseStrategy, Signal

# EMA 200 needs warmup: require 210 bars (CANDLE_LIMIT is 300 in v5.7)
MIN_BARS = 210
FRESH_CROSS_BARS = 5


class TripleConfluenceTrendStrategy(BaseStrategy):
    """Composite #4: ride extended trend waves with triple-class confluence."""
    name = "triple_confluence_trend"

    def analyze(self, df: pd.DataFrame, symbol: str,
                multi_tf_data: Optional[Dict[str, pd.DataFrame]] = None,
                order_book: Optional[Dict] = None) -> Signal:
        if len(df) < MIN_BARS:
            return self._neutral(
                f"Insufficient data for EMA200 ({len(df)}/{MIN_BARS} bars)")

        close = df["close"]
        volume = df["volume"]

        # === Signal Stack: one indicator per class ===
        ema200 = ema(close, 200)
        ema50 = ema(close, 50)
        rsi_series = rsi(close, 14)
        obv_series = obv(close, volume)
        vol_avg = volume_sma(volume, 20)

        price = float(close.iloc[-1])
        e50 = float(ema50.iloc[-1])
        e200 = float(ema200.iloc[-1])
        rsi_now = float(rsi_series.iloc[-1])
        rsi_prev = float(rsi_series.iloc[-2])
        vol_ratio = float(volume.iloc[-1] / max(float(vol_avg.iloc[-1]), 1e-9))
        obv_rising = float(obv_series.iloc[-1]) > float(obv_series.iloc[-6])

        details = {
            "stack": {"direction": "EMA200+EMA50", "momentum": "RSI14",
                      "liquidity": "Volume/OBV"},
            "price": price, "ema50": e50, "ema200": e200,
            "rsi": rsi_now, "volume_ratio": vol_ratio,
            "obv_rising": obv_rising,
        }

        # === EXIT SIDE (emitted as bearish -> feeds opposite-signal exits) ===
        bearish_div = detect_divergence(close, rsi_series, lookback=50)
        if price < e50 and e50 > e200:
            return self._bear(
                58, f"EMA50 breakdown ({price:.4f} < EMA50 {e50:.4f}) - "
                    f"trend wave broken", details)
        if rsi_now > 70 and bearish_div == "bearish":
            return self._bear(
                60, f"RSI overbought ({rsi_now:.1f}) + bearish divergence - "
                    f"exhaustion", details)

        # === BUY CHECKLIST (Signal Stack) ===
        score = 0
        reasons = []

        # --- DIRECTION class (mandatory gate) ---
        # 1) price above EMA 200
        if price <= e200:
            return self._neutral(
                f"Price below EMA200 ({price:.4f} <= {e200:.4f})", details)
        score += 25
        reasons.append(f"Price above EMA200 ({e200:.4f})")

        # 2) EMA50 > EMA200 (golden-cross state / fresh cross bonus)
        if e50 > e200:
            cross_bars_ago = self._bars_since_golden_cross(ema50, ema200)
            if cross_bars_ago is not None and cross_bars_ago <= FRESH_CROSS_BARS:
                score += 30
                reasons.append(
                    f"FRESH golden cross (EMA50 x EMA200, {cross_bars_ago} bars ago)")
            else:
                score += 20
                reasons.append("EMA50 above EMA200 (uptrend confirmed)")
        else:
            return self._neutral("No golden cross (EMA50 <= EMA200)", details)

        # --- MOMENTUM class ---
        # 3) RSI crossed above 50 (fresh) or holds above it; never > 70
        if rsi_now > 70:
            return self._neutral(
                f"RSI overbought ({rsi_now:.1f} > 70) - do not chase", details)
        if rsi_prev <= 50 < rsi_now:
            score += 25
            reasons.append(f"RSI crossed above 50 ({rsi_prev:.1f} -> {rsi_now:.1f})")
        elif rsi_now > 50:
            score += 12
            reasons.append(f"RSI holding above 50 ({rsi_now:.1f})")
        else:
            return self._neutral(f"RSI below 50 ({rsi_now:.1f})", details)

        # --- LIQUIDITY class ---
        # 4) volume above its 20-bar average
        if vol_ratio >= 1.3:
            score += 20
            reasons.append(f"Strong volume ({vol_ratio:.2f}x avg20)")
        elif vol_ratio >= 1.0:
            score += 15
            reasons.append(f"Volume above average ({vol_ratio:.2f}x)")
        else:
            return self._neutral(
                f"No liquidity confirmation ({vol_ratio:.2f}x avg20)", details)

        # bonus: OBV accumulation
        if obv_rising:
            score += 10
            reasons.append("OBV rising (accumulation)")

        score = max(-100, min(100, score))

        if score >= 60:
            return self._bull(score, reasons, details)
        elif score >= 40:
            return self._bull(score * 0.5,
                              [f"Partial trend signal: {r}" for r in reasons],
                              details)
        return self._neutral(f"Score too low ({score})", details)

    @staticmethod
    def _bars_since_golden_cross(ema50: pd.Series, ema200: pd.Series) -> Optional[int]:
        """Bars since the most recent EMA50 x EMA200 upward cross (None if
        the cross is older than the lookback window or never happened)."""
        diff = (ema50 - ema200).dropna()
        if len(diff) < 2:
            return None
        tail = diff.iloc[-(FRESH_CROSS_BARS + 1):]
        crossed = (tail.shift(1) <= 0) & (tail > 0)
        if crossed.any():
            # position of the LAST True within the tail window
            return int(len(tail) - 1 - crossed.values[::-1].argmax())
        return None
