"""
Composite Strategy #2: LIQUIDITY SWEEP REVERSAL
The Smart Trader's Approach:

"Institutions hunt stop losses. Join them, don't fight them."

A real trader doesn't try to catch falling knives. They:
  1. Wait for price to make a NEW LOW (sweeping stop losses)
  2. Confirm the sweep FAILED (price closed back above the swept level)
  3. Look for additional confluence (RSI oversold, Order Block, FVG)
  4. Confirm with reversal candle (Hammer, Bullish Engulfing)
  5. Enter with SL below the sweep low (tight, low-risk)

This is the ICT methodology used by professional traders.
High win rate because institutions create these patterns to fill orders.

CONFLUENCE CHECKLIST (need 3/5 to trigger):
  ☐ Liquidity Sweep: price made new low then closed back above
  ☐ RSI oversold (< 35) at the time of sweep
  ☐ Bullish reversal candle (Hammer, Bullish Engulfing, Morning Star)
  ☐ Volume spike during sweep (capitulation selling)
  ☐ Wyckoff Spring or Order Block near the sweep low
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
from src.indicators.technical import rsi, macd, atr, bollinger_bands, ema
from src.indicators.patterns import (
    detect_hammer, detect_engulfing, detect_morning_star,
    detect_tweezer_bottom, detect_inverted_hammer
)
from src.indicators.ict import (
    detect_liquidity_sweep, find_order_blocks, find_fair_value_gaps
)
from src.indicators.proprietary import wyckoff_spring, volume_climax
from src.strategies.base import BaseStrategy, Signal


class LiquiditySweepReversalStrategy(BaseStrategy):
    """Composite #2: Liquidity Sweep Reversal — join institutions at stop hunts."""
    name = "liquidity_sweep_reversal"
    weight = 2.0

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
        rsi_val = float(rsi(close, 14).iloc[-1])
        atr_val = float(atr(high, low, close, 14).iloc[-1])
        bb = bollinger_bands(close, 20, 2)
        bb_pct = float(bb["percent_b"].iloc[-1])
        vol_sma = volume.rolling(20).mean()
        vol_ratio = float(volume.iloc[-1] / max(vol_sma.iloc[-1], 1e-9))

        # === CONFLUENCE CHECKLIST ===
        score = 0
        reasons = []
        details = {
            "rsi": rsi_val, "atr": atr_val,
            "bb_percent_b": bb_pct, "volume_ratio": vol_ratio,
        }

        # 1) Liquidity Sweep (the core pattern)
        sweep = detect_liquidity_sweep(df, lookback=30)
        if sweep.get("signal") == "bullish":
            strength = sweep.get("strength", 0.5)
            score += int(strength * 50)
            reasons.append(sweep.get("reason", ""))
            details["liquidity_sweep"] = sweep.get("details", {})
        else:
            return self._neutral("No bullish liquidity sweep detected", details)

        # 2) RSI oversold (< 35)
        if rsi_val < 25:
            score += 25
            reasons.append(f"RSI deeply oversold ({rsi_val:.1f})")
        elif rsi_val < 35:
            score += 18
            reasons.append(f"RSI oversold ({rsi_val:.1f})")
        elif rsi_val < 45:
            score += 8
            reasons.append(f"RSI weak ({rsi_val:.1f})")

        # 3) Bullish reversal candle
        last = df.iloc[-1]
        prev = df.iloc[-2]
        prev2 = df.iloc[-3]
        pattern_found = False
        pattern_name = ""

        if detect_hammer(last["open"], last["high"], last["low"], last["close"]):
            pattern_found = True
            pattern_name = "Hammer"
            score += 18
        elif detect_engulfing(prev["open"], prev["high"], prev["low"], prev["close"],
                              last["open"], last["high"], last["low"], last["close"]) == "bullish":
            pattern_found = True
            pattern_name = "Bullish Engulfing"
            score += 20
        elif detect_morning_star(prev2["open"], prev2["high"], prev2["low"], prev2["close"],
                                   prev["open"], prev["high"], prev["low"], prev["close"],
                                   last["open"], last["high"], last["low"], last["close"]):
            pattern_found = True
            pattern_name = "Morning Star"
            score += 22
        elif detect_tweezer_bottom(prev["open"], prev["high"], prev["low"], prev["close"],
                                     last["open"], last["high"], last["low"], last["close"]):
            pattern_found = True
            pattern_name = "Tweezer Bottom"
            score += 15
        elif detect_inverted_hammer(last["open"], last["high"], last["low"], last["close"]):
            pattern_found = True
            pattern_name = "Inverted Hammer"
            score += 12

        if pattern_found:
            reasons.append(f"Reversal candle: {pattern_name}")

        # 4) Volume spike during sweep (capitulation)
        climax = volume_climax(df, lookback=50, spike_threshold=2.0, drop_threshold=-1.0)
        if climax.get("signal") == "bullish":
            score += 20
            reasons.append(climax.get("reason", "Volume climax (capitulation)"))
        elif vol_ratio > 1.5:
            score += 10
            reasons.append(f"High volume ({vol_ratio:.2f}x avg)")

        # 5) Wyckoff Spring (false breakdown + recovery)
        spring = wyckoff_spring(df, support_lookback=50)
        if spring.get("signal") == "bullish":
            strength = spring.get("strength", 0.5)
            score += int(strength * 25)
            reasons.append(spring.get("reason", "Wyckoff Spring detected"))

        # Bonus: Order Block near current price
        obs = find_order_blocks(df, lookback=50)
        if obs:
            recent_ob = obs[-1]
            if recent_ob["type"] == "bullish":
                current = float(close.iloc[-1])
                if recent_ob["low"] <= current <= recent_ob["high"] * 1.01:
                    score += 15
                    reasons.append(f"Bullish Order Block at {recent_ob['low']:.4f}-{recent_ob['high']:.4f}")

        # Bonus: Bollinger %B < 0.1 (price at/below lower band)
        if bb_pct < 0.05:
            score += 12
            reasons.append(f"Price below lower BB (Percent B={bb_pct:.2f})")
        elif bb_pct < 0.15:
            score += 6
            reasons.append(f"Price near lower BB (Percent B={bb_pct:.2f})")

        # Clamp
        score = max(-100, min(100, score))

        # Need at least 50 points (3/5 checklist + 1 bonus)
        if score >= 50:
            return self._bull(score, reasons, details)
        elif score >= 30:
            return self._bull(score * 0.5, [f"Partial: {r}" for r in reasons], details)
        return self._neutral(f"Confluence too weak ({score})", details)
