"""
Ichimoku Kinko Hyo (v4) — Trend Regime Layer

The Ichimoku cloud acts as the bot's REGIME GATE: it answers
"what market are we in?" before any entry is considered.

  - Price above the cloud + Tenkan > Kijun + green future cloud
    + Chikou above price  ->  BULLISH regime (longs allowed, boosted)
  - Price below the cloud with the mirror conditions  ->  BEARISH regime
    (shorts allowed, longs vetoed)
  - Price INSIDE the cloud  ->  NEUTRAL regime (choppy market: signals
    are discounted, never boosted)

Composite score (-100 .. +100):
    price vs cloud       +/- 25
    tenkan vs kijun      +/- 15
    cloud color (A vs B) +/- 10
    chikou vs price-26   +/- 15
    price vs kijun       +/- 10
    fresh TK cross (<5b) +/- 10

Regime thresholds: >= +30 bullish, <= -30 bearish, else neutral.
"""
import pandas as pd
from typing import Dict

# Standard Ichimoku parameters
TENKAN_PERIOD = 9
KIJUN_PERIOD = 26
SENKOU_B_PERIOD = 52
DISPLACEMENT = 26

# Minimum bars required for a fully-defined cloud
MIN_BARS = SENKOU_B_PERIOD + DISPLACEMENT + 1  # 79

# Regime thresholds on the composite score
REGIME_BULL_THRESHOLD = 30.0
REGIME_BEAR_THRESHOLD = -30.0

# Component weights of the composite score
W_CLOUD_POSITION = 25.0
W_TK_STATE = 15.0
W_CLOUD_COLOR = 10.0
W_CHIKOU = 15.0
W_KIJUN_POSITION = 10.0
W_TK_CROSS = 10.0

# How many recent bars to scan for a fresh Tenkan/Kijun cross
TK_CROSS_WINDOW = 5


def ichimoku_lines(df: pd.DataFrame, tenkan: int = TENKAN_PERIOD,
                   kijun: int = KIJUN_PERIOD, senkou_b: int = SENKOU_B_PERIOD,
                   displacement: int = DISPLACEMENT) -> pd.DataFrame:
    """
    Compute the five Ichimoku lines.

    Tenkan-sen  : (highest high + lowest low) / 2 over `tenkan` bars
    Kijun-sen   : same over `kijun` bars
    Senkou A    : (tenkan + kijun) / 2, shifted `displacement` forward
    Senkou B    : (highest high + lowest low) / 2 over `senkou_b`,
                  shifted `displacement` forward
    Chikou      : close shifted `displacement` back (plotted behind)

    The cloud AT bar t is formed by span_a / span_b at t (already shifted),
    so span_a > span_b at t means a green (bullish) cloud at t.
    """
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    tenkan_sen = (high.rolling(tenkan).max() + low.rolling(tenkan).min()) / 2
    kijun_sen = (high.rolling(kijun).max() + low.rolling(kijun).min()) / 2
    span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)
    span_b = (
        (high.rolling(senkou_b).max() + low.rolling(senkou_b).min()) / 2
    ).shift(displacement)
    chikou = close.shift(-displacement)

    return pd.DataFrame({
        "tenkan": tenkan_sen,
        "kijun": kijun_sen,
        "span_a": span_a,
        "span_b": span_b,
        "chikou": chikou,
    })


def ichimoku_state(df: pd.DataFrame) -> Dict:
    """
    Composite Ichimoku regime analysis for the CURRENT bar.

    Returns {} when there is not enough history (needs MIN_BARS),
    otherwise:
      regime          "bullish" | "bearish" | "neutral"
      score           -100..+100 composite
      price_vs_cloud  "above" | "below" | "inside"
      cloud_color     "green" | "red"
      tk_state        "bullish" | "bearish"
      chikou_state    "bullish" | "bearish"
      price_vs_kijun  "above" | "below"
      tk_cross_recent "bullish" | "bearish" | "none"
      tenkan / kijun / cloud_top / cloud_bottom
      kijun_distance_pct   (signed % distance of price from Kijun)
    """
    if df is None or len(df) < MIN_BARS:
        return {}

    lines = ichimoku_lines(df)
    last = lines.iloc[-1]
    if any(pd.isna(last[k]) for k in ("tenkan", "kijun", "span_a", "span_b")):
        return {}

    close = float(df["close"].iloc[-1])
    tenkan = float(last["tenkan"])
    kijun = float(last["kijun"])
    span_a = float(last["span_a"])
    span_b = float(last["span_b"])
    cloud_top = max(span_a, span_b)
    cloud_bottom = min(span_a, span_b)

    # --- Price vs cloud ---
    if close > cloud_top:
        price_vs_cloud = "above"
    elif close < cloud_bottom:
        price_vs_cloud = "below"
    else:
        price_vs_cloud = "inside"

    # --- Cloud color (future kumo direction) ---
    cloud_color = "green" if span_a > span_b else "red"

    # --- Tenkan / Kijun relationship ---
    tk_state = "bullish" if tenkan > kijun else "bearish"

    # --- Chikou: current close vs close 26 bars ago ---
    chikou_ref = float(df["close"].iloc[-1 - DISPLACEMENT])
    chikou_state = "bullish" if close > chikou_ref else "bearish"

    # --- Price vs Kijun ---
    price_vs_kijun = "above" if close > kijun else "below"

    # --- Fresh Tenkan/Kijun cross within the last few bars ---
    tk_diff = (lines["tenkan"] - lines["kijun"]).iloc[-(TK_CROSS_WINDOW + 1):]
    cross = "none"
    prev = tk_diff.iloc[0]
    for v in tk_diff.iloc[1:]:
        if prev <= 0 < v:
            cross = "bullish"
        elif prev >= 0 > v:
            cross = "bearish"
        prev = v

    # --- Composite score ---
    score = 0.0
    score += {"above": W_CLOUD_POSITION, "below": -W_CLOUD_POSITION,
              "inside": 0.0}[price_vs_cloud]
    score += W_TK_STATE if tk_state == "bullish" else -W_TK_STATE
    score += W_CLOUD_COLOR if cloud_color == "green" else -W_CLOUD_COLOR
    score += W_CHIKOU if chikou_state == "bullish" else -W_CHIKOU
    score += W_KIJUN_POSITION if price_vs_kijun == "above" else -W_KIJUN_POSITION
    if cross == "bullish":
        score += W_TK_CROSS
    elif cross == "bearish":
        score -= W_TK_CROSS
    score = max(-100.0, min(100.0, score))

    if score >= REGIME_BULL_THRESHOLD:
        regime = "bullish"
    elif score <= REGIME_BEAR_THRESHOLD:
        regime = "bearish"
    else:
        regime = "neutral"

    return {
        "regime": regime,
        "score": float(score),
        "price_vs_cloud": price_vs_cloud,
        "cloud_color": cloud_color,
        "tk_state": tk_state,
        "chikou_state": chikou_state,
        "price_vs_kijun": price_vs_kijun,
        "tk_cross_recent": cross,
        "tenkan": tenkan,
        "kijun": kijun,
        "cloud_top": float(cloud_top),
        "cloud_bottom": float(cloud_bottom),
        "kijun_distance_pct": float((close - kijun) / kijun * 100) if kijun else 0.0,
    }
