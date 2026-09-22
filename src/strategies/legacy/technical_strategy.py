"""
Technical Analysis Strategy
Combines RSI, MACD, Bollinger Bands, EMA cross, ADX, Stochastic.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
from src.indicators import technical as ta
from src.strategies.base import BaseStrategy, Signal


class TechnicalStrategy(BaseStrategy):
    name = "technical_analysis"
    weight = 1.5  # heavier weight - core strategy

    def analyze(self, df: pd.DataFrame, symbol: str,
                multi_tf_data: Optional[Dict[str, pd.DataFrame]] = None,
                order_book: Optional[Dict] = None) -> Signal:
        if len(df) < 60:
            return self._neutral("Insufficient data")

        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # Compute indicators
        rsi_val = ta.rsi(close, 14).iloc[-1]
        macd_df = ta.macd(close, 12, 26, 9)
        macd_line = macd_df["macd"].iloc[-1]
        macd_signal = macd_df["signal"].iloc[-1]
        macd_hist = macd_df["histogram"].iloc[-1]
        macd_hist_prev = macd_df["histogram"].iloc[-2] if len(macd_df) > 1 else 0
        bb = ta.bollinger_bands(close, 20, 2)
        bb_upper = bb["upper"].iloc[-1]
        bb_lower = bb["lower"].iloc[-1]
        bb_middle = bb["middle"].iloc[-1]
        bb_pct = bb["percent_b"].iloc[-1]
        ema_short = ta.ema(close, 9).iloc[-1]
        ema_mid = ta.ema(close, 21).iloc[-1]
        ema_long = ta.ema(close, 50).iloc[-1]
        adx_df = ta.adx(high, low, close, 14)
        adx_val = adx_df["adx"].iloc[-1]
        plus_di = adx_df["plus_di"].iloc[-1]
        minus_di = adx_df["minus_di"].iloc[-1]
        stoch = ta.stochastic(high, low, close, 14, 3)
        stoch_k = stoch["k"].iloc[-1]
        stoch_d = stoch["d"].iloc[-1]
        atr_val = ta.atr(high, low, close, 14).iloc[-1]

        current_price = close.iloc[-1]
        score = 0.0
        reasons = []
        details = {
            "rsi": float(rsi_val),
            "macd": float(macd_line),
            "macd_signal": float(macd_signal),
            "macd_hist": float(macd_hist),
            "bb_upper": float(bb_upper),
            "bb_middle": float(bb_middle),
            "bb_lower": float(bb_lower),
            "bb_percent_b": float(bb_pct),
            "ema_9": float(ema_short),
            "ema_21": float(ema_mid),
            "ema_50": float(ema_long),
            "adx": float(adx_val),
            "plus_di": float(plus_di),
            "minus_di": float(minus_di),
            "stoch_k": float(stoch_k),
            "stoch_d": float(stoch_d),
            "atr": float(atr_val),
        }

        # ============ RSI ============
        if rsi_val < 30:
            score += 25
            reasons.append(f"RSI oversold ({rsi_val:.1f})")
        elif rsi_val < 45:
            score += 10
            reasons.append(f"RSI below mid ({rsi_val:.1f})")
        elif rsi_val > 70:
            score -= 25
            reasons.append(f"RSI overbought ({rsi_val:.1f})")
        elif rsi_val > 55:
            score -= 10
            reasons.append(f"RSI above mid ({rsi_val:.1f})")

        # ============ MACD ============
        if macd_hist > 0 and macd_hist_prev <= 0:
            score += 20
            reasons.append("MACD bullish crossover")
        elif macd_hist > 0 and macd_hist > macd_hist_prev:
            score += 10
            reasons.append("MACD histogram rising")
        elif macd_hist < 0 and macd_hist_prev >= 0:
            score -= 20
            reasons.append("MACD bearish crossover")
        elif macd_hist < 0 and macd_hist < macd_hist_prev:
            score -= 10
            reasons.append("MACD histogram falling")

        # ============ Bollinger Bands ============
        if bb_pct < 0.05:
            score += 15
            reasons.append("Price near lower Bollinger Band (oversold)")
        elif bb_pct > 0.95:
            score -= 15
            reasons.append("Price near upper Bollinger Band (overbought)")

        # ============ EMA Cross / Trend ============
        if ema_short > ema_mid > ema_long:
            score += 20
            reasons.append("EMA 9 > 21 > 50 (strong uptrend)")
        elif ema_short > ema_mid:
            score += 10
            reasons.append("EMA 9 > 21 (short uptrend)")
        elif ema_short < ema_mid < ema_long:
            score -= 20
            reasons.append("EMA 9 < 21 < 50 (strong downtrend)")
        elif ema_short < ema_mid:
            score -= 10
            reasons.append("EMA 9 < 21 (short downtrend)")

        # ============ ADX (trend strength) ============
        if adx_val > 25:
            if plus_di > minus_di:
                score += 15
                reasons.append(f"Strong uptrend (ADX={adx_val:.1f}, +DI>-DI)")
            else:
                score -= 15
                reasons.append(f"Strong downtrend (ADX={adx_val:.1f}, -DI>+DI)")
        else:
            reasons.append(f"Weak/no trend (ADX={adx_val:.1f})")

        # ============ Stochastic ============
        if stoch_k < 20 and stoch_k > stoch_d:
            score += 15
            reasons.append("Stochastic bullish cross from oversold")
        elif stoch_k > 80 and stoch_k < stoch_d:
            score -= 15
            reasons.append("Stochastic bearish cross from overbought")

        # Clamp
        score = max(-100, min(100, score))

        # Lowered thresholds: was >20, now >8 (much less strict)
        if score > 8:
            return self._bull(score, reasons, details)
        elif score < -8:
            return self._bear(abs(score), reasons, details)
        return self._neutral(f"Neutral - mixed signals ({score:+.1f})", details)
