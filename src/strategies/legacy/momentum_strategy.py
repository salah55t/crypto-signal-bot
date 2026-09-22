"""
Momentum & Trend Strategy
Multi-timeframe trend alignment + momentum confirmation.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
from src.indicators import technical as ta
from src.strategies.base import BaseStrategy, Signal


class MomentumStrategy(BaseStrategy):
    name = "momentum_trend"
    weight = 1.4

    def analyze(self, df: pd.DataFrame, symbol: str,
                multi_tf_data: Optional[Dict[str, pd.DataFrame]] = None,
                order_book: Optional[Dict] = None) -> Signal:
        if len(df) < 60:
            return self._neutral("Insufficient data")

        close = df["close"]
        high = df["high"]
        low = df["low"]

        # Indicators
        roc_12 = ta.roc(close, 12).iloc[-1]
        momentum_10 = ta.momentum(close, 10).iloc[-1]
        rsi_val = ta.rsi(close, 14).iloc[-1]
        cci_val = ta.cci(high, low, close, 20).iloc[-1]
        williams = ta.williams_r(high, low, close, 14).iloc[-1]
        ema_50 = ta.ema(close, 50).iloc[-1]
        ema_200_val = ta.ema(close, 200).iloc[-1] if len(close) > 200 else None
        trend = ta.detect_trend(close, 20, 50).iloc[-1]

        # Detect divergence (price vs RSI)
        div = ta.detect_divergence(close, ta.rsi(close, 14), lookback=50)

        score = 0.0
        reasons = []
        details = {
            "roc_12": float(roc_12),
            "momentum_10": float(momentum_10),
            "rsi": float(rsi_val),
            "cci": float(cci_val),
            "williams_r": float(williams),
            "ema_50": float(ema_50),
            "trend": int(trend),
            "divergence": div,
        }

        # ============ ROC ============
        if roc_12 > 5:
            score += 20
            reasons.append(f"Strong positive ROC ({roc_12:.2f}%)")
        elif roc_12 > 1:
            score += 10
            reasons.append(f"Positive ROC ({roc_12:.2f}%)")
        elif roc_12 < -5:
            score -= 20
            reasons.append(f"Strong negative ROC ({roc_12:.2f}%)")
        elif roc_12 < -1:
            score -= 10
            reasons.append(f"Negative ROC ({roc_12:.2f}%)")

        # ============ CCI ============
        if cci_val < -100:
            score += 15
            reasons.append(f"CCI oversold ({cci_val:.1f})")
        elif cci_val > 100:
            score -= 15
            reasons.append(f"CCI overbought ({cci_val:.1f})")

        # ============ Williams %R ============
        if williams > -20:
            score -= 10
            reasons.append(f"Williams %R overbought ({williams:.1f})")
        elif williams < -80:
            score += 10
            reasons.append(f"Williams %R oversold ({williams:.1f})")

        # ============ EMA Trend ============
        if ema_200_val:
            details["ema_200"] = float(ema_200_val)
            if close.iloc[-1] > ema_50 > ema_200_val:
                score += 20
                reasons.append("Price > EMA50 > EMA200 (bullish structure)")
            elif close.iloc[-1] < ema_50 < ema_200_val:
                score -= 20
                reasons.append("Price < EMA50 < EMA200 (bearish structure)")
        else:
            if trend == 1:
                score += 15
                reasons.append("EMA trend up (short above long)")
            elif trend == -1:
                score -= 15
                reasons.append("EMA trend down (short below long)")

        # ============ Divergence ============
        if div == "bullish":
            score += 25
            reasons.append("Bullish divergence (price lower low, RSI higher low)")
        elif div == "bearish":
            score -= 25
            reasons.append("Bearish divergence (price higher high, RSI lower high)")

        # ============ Multi-timeframe confirmation ============
        if multi_tf_data:
            aligned_bull = 0
            aligned_bear = 0
            for tf, tf_df in multi_tf_data.items():
                if len(tf_df) < 60:
                    continue
                tf_trend = ta.detect_trend(tf_df["close"], 20, 50).iloc[-1]
                if tf_trend == 1:
                    aligned_bull += 1
                elif tf_trend == -1:
                    aligned_bear += 1
            total_tf = len(multi_tf_data)
            if total_tf > 0:
                if aligned_bull == total_tf:
                    score += 20
                    reasons.append(f"Multi-timeframe trend ALL bullish ({aligned_bull}/{total_tf})")
                elif aligned_bear == total_tf:
                    score -= 20
                    reasons.append(f"Multi-timeframe trend ALL bearish ({aligned_bear}/{total_tf})")
                elif aligned_bull > aligned_bear:
                    score += 5
                    reasons.append(f"Multi-timeframe mostly bullish ({aligned_bull}/{total_tf})")
                elif aligned_bear > aligned_bull:
                    score -= 5
                    reasons.append(f"Multi-timeframe mostly bearish ({aligned_bear}/{total_tf})")

        score = max(-100, min(100, score))

        # Lowered thresholds: was >20, now >8 (much less strict)
        if score > 8:
            return self._bull(score, reasons, details)
        elif score < -8:
            return self._bear(abs(score), reasons, details)
        return self._neutral(f"Momentum neutral ({score:+.1f})", details)
