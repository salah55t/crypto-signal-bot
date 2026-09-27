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
from src.indicators.technical import rsi, macd, atr, bollinger_bands, ema, adx
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
        adx_df = adx(high, low, close, 14)
        adx_val = float(adx_df["adx"].iloc[-1])
        plus_di = float(adx_df["plus_di"].iloc[-1])
        minus_di = float(adx_df["minus_di"].iloc[-1])

        # === CONFLUENCE CHECKLIST ===
        score = 0
        reasons = []
        details = {
            "rsi": rsi_val, "atr": atr_val,
            "bb_percent_b": bb_pct, "volume_ratio": vol_ratio,
            "adx": adx_val,
        }

        # 1) Liquidity Sweep (the core pattern) - BOTH sides (v5.16: the
        #    old code only ever fired bullish, biasing the scorer long in
        #    downtrends and never closing on bearish sweeps).
        sweep = detect_liquidity_sweep(df, lookback=30)
        sweep_dir = sweep.get("signal", "none")
        if sweep_dir not in ("bullish", "bearish"):
            return self._neutral("No liquidity sweep detected", details)
        strength = sweep.get("strength", 0.5)
        score += int(strength * 50)
        reasons.append(sweep.get("reason", ""))
        details["liquidity_sweep"] = sweep.get("details", {})

        # v5.16 KNIFE FILTER: a sweep AGAINST an extremely strong trend is
        # a trap, not a reversal (one-sided books keep running the stops).
        # Block counter-trend sweeps when ADX > 40 and the DIs agree.
        if sweep_dir == "bullish" and adx_val > 40 and minus_di > plus_di * 1.4:
            return self._neutral(
                f"Bullish sweep against a violent downtrend "
                f"(ADX={adx_val:.1f}) - knife, not reversal", details)
        if sweep_dir == "bearish" and adx_val > 40 and plus_di > minus_di * 1.4:
            return self._neutral(
                f"Bearish sweep against a violent uptrend "
                f"(ADX={adx_val:.1f}) - knife, not reversal", details)

        # 2) RSI at the sweep extreme (direction-aware now)
        if sweep_dir == "bullish":
            if rsi_val < 25:
                score += 25
                reasons.append(f"RSI deeply oversold ({rsi_val:.1f})")
            elif rsi_val < 35:
                score += 18
                reasons.append(f"RSI oversold ({rsi_val:.1f})")
            elif rsi_val < 45:
                score += 8
                reasons.append(f"RSI weak ({rsi_val:.1f})")
        else:
            if rsi_val > 75:
                score += 25
                reasons.append(f"RSI deeply overbought ({rsi_val:.1f})")
            elif rsi_val > 65:
                score += 18
                reasons.append(f"RSI overbought ({rsi_val:.1f})")
            elif rsi_val > 55:
                score += 8
                reasons.append(f"RSI elevated ({rsi_val:.1f})")

        # 3) Reversal candle (direction-aware)
        last = df.iloc[-1]
        prev = df.iloc[-2]
        prev2 = df.iloc[-3]
        pattern_found = False
        pattern_name = ""

        if sweep_dir == "bullish":
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
        else:
            from src.indicators.patterns import (
                detect_shooting_star, detect_evening_star,
                detect_dark_cloud, detect_three_black_crows,
            )
            if detect_shooting_star(last["open"], last["high"], last["low"], last["close"]):
                pattern_found = True
                pattern_name = "Shooting Star"
                score += 18
            elif detect_engulfing(prev["open"], prev["high"], prev["low"], prev["close"],
                                  last["open"], last["high"], last["low"], last["close"]) == "bearish":
                pattern_found = True
                pattern_name = "Bearish Engulfing"
                score += 20
            elif detect_evening_star(prev2["open"], prev2["high"], prev2["low"], prev2["close"],
                                     prev["open"], prev["high"], prev["low"], prev["close"],
                                     last["open"], last["high"], last["low"], last["close"]):
                pattern_found = True
                pattern_name = "Evening Star"
                score += 22
            elif detect_dark_cloud(prev["open"], prev["high"], prev["low"], prev["close"],
                                   last["open"], last["high"], last["low"], last["close"]):
                pattern_found = True
                pattern_name = "Dark Cloud Cover"
                score += 18
            elif detect_three_black_crows(prev2["open"], prev2["high"], prev2["low"], prev2["close"],
                                          prev["open"], prev["high"], prev["low"], prev["close"],
                                          last["open"], last["high"], last["low"], last["close"]):
                pattern_found = True
                pattern_name = "Three Black Crows"
                score += 15

        if pattern_found:
            reasons.append(f"Reversal candle: {pattern_name}")

        # 4) Volume spike during sweep (capitulation)
        climax = volume_climax(df, lookback=50, spike_threshold=2.0, drop_threshold=-1.0)
        if climax.get("signal") == sweep_dir:
            score += 20
            reasons.append(climax.get("reason", "Volume climax"))
        elif vol_ratio > 1.5:
            score += 10
            reasons.append(f"High volume ({vol_ratio:.2f}x avg)")

        # 5) Wyckoff Spring (bullish) / Upthrust (bearish false break)
        spring = wyckoff_spring(df, support_lookback=50)
        if spring.get("signal") == sweep_dir:
            strength = spring.get("strength", 0.5)
            score += int(strength * 25)
            reasons.append(spring.get(
                "reason",
                "Wyckoff Spring" if sweep_dir == "bullish" else "Wyckoff Upthrust"))

        # Bonus: Order Block near current price (direction-aware)
        obs = find_order_blocks(df, lookback=50)
        if obs:
            recent_ob = obs[-1]
            current = float(close.iloc[-1])
            if sweep_dir == "bullish" and recent_ob["type"] == "bullish":
                if recent_ob["low"] <= current <= recent_ob["high"] * 1.01:
                    score += 15
                    reasons.append(f"Bullish Order Block at {recent_ob['low']:.4f}-{recent_ob['high']:.4f}")
            elif sweep_dir == "bearish" and recent_ob["type"] == "bearish":
                if recent_ob["low"] * 0.99 <= current <= recent_ob["high"]:
                    score += 15
                    reasons.append(f"Bearish Order Block at {recent_ob['low']:.4f}-{recent_ob['high']:.4f}")

        # Bonus: Bollinger extreme on the sweep side
        if sweep_dir == "bullish":
            if bb_pct < 0.05:
                score += 12
                reasons.append(f"Price below lower BB (Percent B={bb_pct:.2f})")
            elif bb_pct < 0.15:
                score += 6
                reasons.append(f"Price near lower BB (Percent B={bb_pct:.2f})")
        else:
            if bb_pct > 0.95:
                score += 12
                reasons.append(f"Price above upper BB (Percent B={bb_pct:.2f})")
            elif bb_pct > 0.85:
                score += 6
                reasons.append(f"Price near upper BB (Percent B={bb_pct:.2f})")

        # Clamp
        score = max(-100, min(100, score))

        # Need at least 50 points (3/5 checklist + 1 bonus)
        if score >= 50:
            sig = self._bull(score, reasons, details) if sweep_dir == "bullish" \
                else self._bear(score, reasons, details)
            return sig
        elif score >= 30:
            scaled = score * 0.5
            partial = [f"Partial: {r}" for r in reasons]
            return self._bull(scaled, partial, details) if sweep_dir == "bullish" \
                else self._bear(scaled, partial, details)
        return self._neutral(f"Confluence too weak ({score})", details)
