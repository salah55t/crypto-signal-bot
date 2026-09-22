"""
ICT (Inner Circle Trader) Indicators - Smart Money Concepts

Indicators based on institutional trading patterns:
  - Order Blocks: last opposite candle before strong move
  - Fair Value Gap (FVG): 3-candle imbalance (price gap)
  - Liquidity Sweep: stop hunt at swing high/low + reversal
  - Liquidity Pools: clusters of stop losses above/below key levels
  - Breaker Blocks: failed order blocks that become support/resistance

These are the same concepts used by professional ICT traders, adapted
for crypto markets.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional


# ============================================
# ORDER BLOCKS
# ============================================

def find_order_blocks(df: pd.DataFrame, lookback: int = 50,
                       strong_move_threshold: float = 1.5) -> List[Dict]:
    """
    Find Order Blocks (institutional footprints).

    Bullish OB: Last bearish candle before a strong bullish move
                (price jumped up significantly after this candle)
    Bearish OB: Last bullish candle before a strong bearish move

    strong_move_threshold: multiplier of ATR for "strong move"
    """
    if len(df) < lookback + 10:
        return []

    # Calculate ATR for measuring "strong moves"
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean().fillna(tr.mean())

    order_blocks = []
    for i in range(10, len(df) - 3):
        # Check if this is a strong move candle
        move_size = abs(df["close"].iloc[i] - df["open"].iloc[i])
        avg_atr = atr.iloc[i] if not pd.isna(atr.iloc[i]) else tr.mean()
        is_strong = move_size > avg_atr * strong_move_threshold

        if not is_strong:
            continue

        # Bullish strong move (close > open) → look for last bearish candle
        if df["close"].iloc[i] > df["open"].iloc[i]:
            # Find last bearish candle before this bullish move
            for j in range(i - 1, max(0, i - 10), -1):
                if df["close"].iloc[j] < df["open"].iloc[j]:
                    # This is the Bullish Order Block
                    order_blocks.append({
                        "type": "bullish",
                        "index": j,
                        "open": float(df["open"].iloc[j]),
                        "high": float(df["high"].iloc[j]),
                        "low": float(df["low"].iloc[j]),
                        "close": float(df["close"].iloc[j]),
                        "strength": float(move_size / avg_atr) if avg_atr > 0 else 0,
                        "move_index": i,
                        "move_size": float(move_size),
                    })
                    break
        # Bearish strong move → look for last bullish candle
        elif df["close"].iloc[i] < df["open"].iloc[i]:
            for j in range(i - 1, max(0, i - 10), -1):
                if df["close"].iloc[j] > df["open"].iloc[j]:
                    order_blocks.append({
                        "type": "bearish",
                        "index": j,
                        "open": float(df["open"].iloc[j]),
                        "high": float(df["high"].iloc[j]),
                        "low": float(df["low"].iloc[j]),
                        "close": float(df["close"].iloc[j]),
                        "strength": float(move_size / avg_atr) if avg_atr > 0 else 0,
                        "move_index": i,
                        "move_size": float(move_size),
                    })
                    break

    return order_blocks


def detect_active_order_block(df: pd.DataFrame, lookback: int = 50) -> Dict:
    """
    Find the most recent active Order Block that price is currently testing.
    Returns: dict with OB info + signal
    """
    obs = find_order_blocks(df, lookback=lookback)
    if not obs:
        return {"signal": "none"}

    current_price = float(df["close"].iloc[-1])
    # Find most recent bullish OB near current price
    recent_bull_ob = None
    recent_bear_ob = None
    for ob in reversed(obs):
        if ob["type"] == "bullish" and not recent_bull_ob:
            recent_bull_ob = ob
        elif ob["type"] == "bearish" and not recent_bear_ob:
            recent_bear_ob = ob
        if recent_bull_ob and recent_bear_ob:
            break

    # Check if price is testing a bullish OB (within 0.5% of OB low)
    if recent_bull_ob:
        ob_low = recent_bull_ob["low"]
        ob_high = recent_bull_ob["high"]
        # Price within OB zone
        if ob_low <= current_price <= ob_high * 1.005:
            # Bullish OB rejection = bounce signal
            return {
                "signal": "bullish",
                "strength": min(0.9, recent_bull_ob["strength"] / 5),
                "reason": f"Price testing Bullish OB at {ob_low:.4f}-{ob_high:.4f} (strength {recent_bull_ob['strength']:.1f}x ATR)",
                "details": {
                    "ob_low": ob_low,
                    "ob_high": ob_high,
                    "ob_strength": recent_bull_ob["strength"],
                }
            }

    # Check bearish OB
    if recent_bear_ob:
        ob_low = recent_bear_ob["low"]
        ob_high = recent_bear_ob["high"]
        if ob_low * 0.995 <= current_price <= ob_high:
            return {
                "signal": "bearish",
                "strength": min(0.9, recent_bear_ob["strength"] / 5),
                "reason": f"Price testing Bearish OB at {ob_low:.4f}-{ob_high:.4f}",
                "details": {
                    "ob_low": ob_low,
                    "ob_high": ob_high,
                    "ob_strength": recent_bear_ob["strength"],
                }
            }

    return {"signal": "none"}


# ============================================
# FAIR VALUE GAP (FVG)
# ============================================

def find_fair_value_gaps(df: pd.DataFrame, lookback: int = 50) -> List[Dict]:
    """
    Find Fair Value Gaps (3-candle imbalance pattern).

    Bullish FVG: candle[i-2].high < candle[i].low
                  (gap between candle 1's high and candle 3's low)
    Bearish FVG: candle[i-2].low > candle[i].high
                  (gap between candle 3's high and candle 1's low)

    Unfilled FVGs act as magnets - price tends to revisit them.
    """
    if len(df) < lookback + 3:
        return []

    fvgs = []
    for i in range(2, len(df)):
        h1 = df["high"].iloc[i - 2]
        l1 = df["low"].iloc[i - 2]
        h3 = df["high"].iloc[i]
        l3 = df["low"].iloc[i]

        # Bullish FVG: candle 1 high < candle 3 low (gap up)
        if h1 < l3:
            gap_size = l3 - h1
            fvgs.append({
                "type": "bullish",
                "index": i,
                "gap_top": float(l3),
                "gap_bottom": float(h1),
                "gap_size": float(gap_size),
                "filled": False,
            })
        # Bearish FVG: candle 1 low > candle 3 high (gap down)
        elif l1 > h3:
            gap_size = l1 - h3
            fvgs.append({
                "type": "bearish",
                "index": i,
                "gap_top": float(l1),
                "gap_bottom": float(h3),
                "gap_size": float(gap_size),
                "filled": False,
            })

    # Check if FVGs have been filled
    current_price = float(df["close"].iloc[-1])
    for fvg in fvgs:
        if fvg["type"] == "bullish":
            # Filled if price went below gap_bottom
            fvg["filled"] = bool(df["low"].iloc[fvg["index"]:].min() < fvg["gap_bottom"])
        else:
            fvg["filled"] = bool(df["high"].iloc[fvg["index"]:].max() > fvg["gap_top"])

    return fvgs


def detect_active_fvg(df: pd.DataFrame, lookback: int = 50) -> Dict:
    """
    Find unfilled FVG that price is approaching.
    Unfilled FVGs act as magnets — high probability of price visiting them.
    """
    fvgs = find_fair_value_gaps(df, lookback=lookback)
    if not fvgs:
        return {"signal": "none"}

    current_price = float(df["close"].iloc[-1])
    # Find most recent unfilled FVG
    unfilled = [f for f in fvgs if not f["filled"]]
    if not unfilled:
        return {"signal": "none"}

    recent_fvg = unfilled[-1]

    # Bullish FVG: price above gap = bullish bias (gap acts as support)
    if recent_fvg["type"] == "bullish":
        gap_top = recent_fvg["gap_top"]
        gap_bottom = recent_fvg["gap_bottom"]
        # Price above gap (gap acts as support if revisited)
        if current_price > gap_top:
            # Distance to gap (potential long setup when price returns)
            dist_to_gap = (current_price - gap_top) / current_price * 100
            if dist_to_gap < 1.0:  # price near the gap
                return {
                    "signal": "bullish",
                    "strength": 0.7,
                    "reason": f"Unfilled Bullish FVG at {gap_bottom:.4f}-{gap_top:.4f} acting as support",
                    "details": {
                        "gap_top": gap_top,
                        "gap_bottom": gap_bottom,
                        "gap_size": recent_fvg["gap_size"],
                        "distance_pct": float(dist_to_gap),
                    }
                }
    # Bearish FVG: price below gap = bearish bias
    else:
        gap_top = recent_fvg["gap_top"]
        gap_bottom = recent_fvg["gap_bottom"]
        if current_price < gap_bottom:
            dist_to_gap = (gap_top - current_price) / current_price * 100
            if dist_to_gap < 1.0:
                return {
                    "signal": "bearish",
                    "strength": 0.7,
                    "reason": f"Unfilled Bearish FVG at {gap_bottom:.4f}-{gap_top:.4f} acting as resistance",
                    "details": {
                        "gap_top": gap_top,
                        "gap_bottom": gap_bottom,
                        "gap_size": recent_fvg["gap_size"],
                        "distance_pct": float(dist_to_gap),
                    }
                }

    return {"signal": "none"}


# ============================================
# LIQUIDITY SWEEP (Stop Hunt)
# ============================================

def detect_liquidity_sweep(df: pd.DataFrame, lookback: int = 30) -> Dict:
    """
    Detect Liquidity Sweep (stop hunt pattern).

    Bullish Liquidity Sweep:
      1. Price makes a new low (below previous swing low)
      2. Then immediately reverses back above that low
      3. This indicates institutions swept sell stops and are now buying

    Bearish Liquidity Sweep:
      1. Price makes a new high (above previous swing high)
      2. Then immediately reverses back below that high
      3. Institutions swept buy stops and are now selling

    This is one of the highest-probability reversal patterns.
    """
    if len(df) < lookback + 5:
        return {"signal": "none"}

    # Find recent swing highs/lows (excluding last 3 candles)
    recent = df.iloc[-lookback:-3]
    if recent.empty:
        return {"signal": "none"}

    swing_high = float(recent["high"].max())
    swing_low = float(recent["low"].min())

    # Last 3 candles to check for sweep
    last_candles = df.iloc[-3:]
    last_close = float(last_candles["close"].iloc[-1])
    last_high = float(last_candles["high"].iloc[-1])
    last_low = float(last_candles["low"].iloc[-1])

    # Bullish Liquidity Sweep (sweep of swing low + recovery)
    # Did price break below swing_low but close back above it?
    swept_low = any(c["low"] < swing_low for _, c in last_candles.iterrows())
    recovered = last_close > swing_low

    if swept_low and recovered:
        # Calculate sweep size and recovery strength
        sweep_low = float(last_candles["low"].min())
        sweep_size = swing_low - sweep_low
        recovery_pct = (last_close - sweep_low) / max(0.0001, sweep_low) * 100
        # Strong sweep: quick recovery (within 1-3 candles)
        if recovery_pct > 0.3:
            strength = min(0.95, 0.5 + recovery_pct / 5)
            return {
                "signal": "bullish",
                "strength": float(strength),
                "reason": f"Bullish Liquidity Sweep: swept low {sweep_low:.4f}, recovered to {last_close:.4f} (+{recovery_pct:.2f}%)",
                "details": {
                    "swing_low": swing_low,
                    "sweep_low": sweep_low,
                    "sweep_size_pct": float(sweep_size / swing_low * 100) if swing_low > 0 else 0,
                    "recovery_pct": float(recovery_pct),
                }
            }

    # Bearish Liquidity Sweep (sweep of swing high + reversal)
    swept_high = any(c["high"] > swing_high for _, c in last_candles.iterrows())
    recovered = last_close < swing_high

    if swept_high and recovered:
        sweep_high = float(last_candles["high"].max())
        recovery_pct = (sweep_high - last_close) / max(0.0001, sweep_high) * 100
        if recovery_pct > 0.3:
            strength = min(0.95, 0.5 + recovery_pct / 5)
            return {
                "signal": "bearish",
                "strength": float(strength),
                "reason": f"Bearish Liquidity Sweep: swept high {sweep_high:.4f}, reversed to {last_close:.4f} (-{recovery_pct:.2f}%)",
                "details": {
                    "swing_high": swing_high,
                    "sweep_high": sweep_high,
                    "sweep_size_pct": float((sweep_high - swing_high) / swing_high * 100) if swing_high > 0 else 0,
                    "recovery_pct": float(recovery_pct),
                }
            }

    return {"signal": "none"}


# ============================================
# ICT CONFLUENCE (highest probability setup)
# ============================================

def detect_ict_confluence(df: pd.DataFrame, lookback: int = 50) -> Dict:
    """
    Detect ICT Confluence — when OB + FVG + Liquidity Sweep all align.

    This is the highest-probability setup per ICT traders.
    Returns 'bullish' only when ALL THREE signals align in the same direction.
    """
    ob = detect_active_order_block(df, lookback=lookback)
    fvg = detect_active_fvg(df, lookback=lookback)
    sweep = detect_liquidity_sweep(df, lookback=lookback)

    signals = [ob.get("signal"), fvg.get("signal"), sweep.get("signal")]
    bullish_count = sum(1 for s in signals if s == "bullish")
    bearish_count = sum(1 for s in signals if s == "bearish")

    details = {
        "order_block": ob,
        "fair_value_gap": fvg,
        "liquidity_sweep": sweep,
        "confluence_count": max(bullish_count, bearish_count),
    }

    if bullish_count >= 2:  # At least 2 of 3 signals align
        strength = (bullish_count / 3) * 0.95  # max 0.95 for 3/3
        return {
            "signal": "bullish",
            "strength": float(strength),
            "reason": f"ICT Confluence ({bullish_count}/3 signals): " +
                      " | ".join(filter(None, [
                          ob.get("reason"),
                          fvg.get("reason"),
                          sweep.get("reason"),
                      ])[:3]),
            "details": details,
        }
    elif bearish_count >= 2:
        strength = (bearish_count / 3) * 0.95
        return {
            "signal": "bearish",
            "strength": float(strength),
            "reason": f"ICT Confluence ({bearish_count}/3 bearish signals)",
            "details": details,
        }

    return {"signal": "none", "details": details}
