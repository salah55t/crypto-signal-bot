"""
Liquidity & Order Book Analysis
Detects buy/sell wall imbalance, support/resistance, and key price levels.
"""
import pandas as pd
import numpy as np
from typing import Dict


def analyze_order_book(ob: Dict) -> Dict:
    """
    Compute order book imbalance metrics.
    Input: dict with 'bids' and 'asks' as DataFrames (price, qty).
    """
    bids = ob.get("bids")
    asks = ob.get("asks")
    if bids is None or asks is None or bids.empty or asks.empty:
        return {}

    total_bid_vol = bids["qty"].sum()
    total_ask_vol = asks["qty"].sum()
    total_vol = total_bid_vol + total_ask_vol

    if total_vol == 0:
        return {}

    imbalance = (total_bid_vol - total_ask_vol) / total_vol
    bid_pct = total_bid_vol / total_vol
    ask_pct = total_ask_vol / total_vol

    # Find largest walls
    bid_wall = bids.loc[bids["qty"].idxmax()]
    ask_wall = asks.loc[asks["qty"].idxmax()]

    # Top 3 bid and ask walls (price levels with significant volume)
    top_bids = bids.nlargest(3, "qty")
    top_asks = asks.nlargest(3, "qty")

    # Spread
    best_bid = bids["price"].max()
    best_ask = asks["price"].min()
    spread = best_ask - best_bid
    spread_pct = spread / best_ask * 100 if best_ask > 0 else 0

    return {
        "best_bid": float(best_bid),
        "best_ask": float(best_ask),
        "spread": float(spread),
        "spread_pct": float(spread_pct),
        "total_bid_volume": float(total_bid_vol),
        "total_ask_volume": float(total_ask_vol),
        "imbalance": float(imbalance),  # +1 = heavy buy, -1 = heavy sell
        "bid_pct": float(bid_pct),
        "ask_pct": float(ask_pct),
        "bid_wall_price": float(bid_wall["price"]),
        "bid_wall_qty": float(bid_wall["qty"]),
        "ask_wall_price": float(ask_wall["price"]),
        "ask_wall_qty": float(ask_wall["qty"]),
        "top_bids": top_bids.to_dict("records"),
        "top_asks": top_asks.to_dict("records"),
        "bias": "bullish" if imbalance > 0.15 else (
            "bearish" if imbalance < -0.15 else "neutral"
        ),
    }


def find_support_resistance(df: pd.DataFrame, lookback: int = 50,
                             min_touches: int = 2) -> Dict:
    """
    Identify local support/resistance levels from swing highs/lows.
    """
    if len(df) < lookback:
        return {"supports": [], "resistances": []}

    window = df.iloc[-lookback:].copy()
    highs = window["high"].values
    lows = window["low"].values
    closes = window["close"].values

    # Find local maxima/minima using simple comparison
    resistances = []
    supports = []
    for i in range(2, len(highs) - 2):
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
           highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            resistances.append(highs[i])
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
           lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            supports.append(lows[i])

    # Cluster nearby levels (within 1%)
    current_price = closes[-1]

    def cluster(levels, tol=0.01):
        if not levels:
            return []
        levels = sorted(levels)
        clusters = [[levels[0]]]
        for lvl in levels[1:]:
            if abs(lvl - clusters[-1][-1]) / clusters[-1][-1] < tol:
                clusters[-1].append(lvl)
            else:
                clusters.append([lvl])
        return [sum(c) / len(c) for c in clusters if len(c) >= 1]

    res_clusters = cluster(resistances)
    sup_clusters = cluster(supports)

    # Filter: only keep levels above (resistance) or below (support) current
    # price. Sort NEAREST-first: resistances ascending (closest above price),
    # supports descending (closest below price). The old code kept supports
    # in ascending order, so "nearest_support" actually returned the FARTHEST
    # (deepest) support - fixed in v3.
    resistances = sorted([r for r in res_clusters if r > current_price])[:3]
    supports = sorted([s for s in sup_clusters if s < current_price], reverse=True)[:3]

    return {
        "current_price": float(current_price),
        "supports": [float(s) for s in supports],
        "resistances": [float(r) for r in resistances],
        "nearest_support": float(supports[0]) if supports else None,
        "nearest_resistance": float(resistances[0]) if resistances else None,
    }


def liquidity_summary(df: pd.DataFrame, ob: Dict) -> Dict:
    """Combined liquidity picture for a symbol."""
    sr = find_support_resistance(df, lookback=50)
    ob_metrics = analyze_order_book(ob)
    return {
        "support_resistance": sr,
        "order_book": ob_metrics,
        "bias": ob_metrics.get("bias", "neutral"),
        "imbalance": ob_metrics.get("imbalance", 0),
    }
