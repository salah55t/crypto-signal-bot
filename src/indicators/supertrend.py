"""SuperTrend indicator - exact TradingView algorithm.

The user's v5.22 double-indicator strategy specifies SuperTrend with
ATR length 2 and multiplier 2. This module implements the canonical
version (identical semantics to the Pine `ta.supertrend`):

    atr      = Wilder RMA of True Range (src.indicators.technical.atr)
    up_basic = hl2 - mult * atr      (support line in an uptrend)
    dn_basic = hl2 + mult * atr      (resistance line in a downtrend)
    up = close[1] > up[1]_final ? max(up_basic, up[1]_final) : up_basic
    dn = close[1] < dn[1]_final ? min(dn_basic, dn[1]_final) : dn_basic
    trend = -1 -> +1 when close crosses above the previous dn band
            +1 -> -1 when close crosses below the previous up band
    st_line = trend == 1 ? up : dn

`src.indicators.technical.atr` already uses Wilder smoothing
(ewm alpha=1/period), so the bands match TradingView exactly.
"""
from typing import Tuple

import numpy as np
import pandas as pd

from src.indicators.technical import atr


def supertrend(high: pd.Series, low: pd.Series, close: pd.Series,
               period: int = 2, multiplier: float = 2.0
               ) -> Tuple[pd.Series, pd.Series]:
    """Return (st_line, trend) where trend is +1 (uptrend) / -1 (downtrend).

    `st_line` is the plotted SuperTrend value: the trailed support below
    price in an uptrend, the trailed resistance above price in a downtrend.
    """
    h = pd.to_numeric(high, errors="coerce").reset_index(drop=True)
    l = pd.to_numeric(low, errors="coerce").reset_index(drop=True)
    c = pd.to_numeric(close, errors="coerce").reset_index(drop=True)
    n = len(c)
    if n == 0:
        empty = pd.Series(dtype=float)
        return empty, pd.Series(dtype=float)

    atr_vals = atr(h, l, c, period).to_numpy(dtype=float)
    hl2 = ((h + l) / 2.0).to_numpy(dtype=float)

    up_basic = hl2 - multiplier * atr_vals
    dn_basic = hl2 + multiplier * atr_vals

    up_final = np.full(n, np.nan)
    dn_final = np.full(n, np.nan)
    trend = np.ones(n, dtype=int)
    st_line = np.full(n, np.nan)

    for i in range(n):
        if np.isnan(atr_vals[i]):
            # warmup bars (before the first full ATR window): no bands yet
            trend[i] = 1
            continue
        prev_close = c.iloc[i - 1] if i > 0 else np.nan

        # trail the support band
        if i == 0 or np.isnan(up_final[i - 1]):
            up_final[i] = up_basic[i]
        else:
            up_final[i] = max(up_basic[i], up_final[i - 1]) \
                if (not np.isnan(prev_close) and prev_close > up_final[i - 1]) \
                else up_basic[i]
        # trail the resistance band
        if i == 0 or np.isnan(dn_final[i - 1]):
            dn_final[i] = dn_basic[i]
        else:
            dn_final[i] = min(dn_basic[i], dn_final[i - 1]) \
                if (not np.isnan(prev_close) and prev_close < dn_final[i - 1]) \
                else dn_basic[i]

        prev_trend = trend[i - 1] if i > 0 else 1
        prev_dn = dn_final[i - 1] if i > 0 else np.nan
        prev_up = up_final[i - 1] if i > 0 else np.nan

        if prev_trend == -1 and (np.isnan(prev_dn) or c.iloc[i] > prev_dn):
            trend[i] = 1
        elif prev_trend == 1 and (np.isnan(prev_up) or c.iloc[i] < prev_up):
            trend[i] = -1
        else:
            trend[i] = prev_trend

        st_line[i] = up_final[i] if trend[i] == 1 else dn_final[i]

    st = pd.Series(st_line, index=close.index)
    tr = pd.Series(trend, index=close.index)
    return st, tr


def st_trend_series(high: pd.Series, low: pd.Series, close: pd.Series,
                    period: int = 2, multiplier: float = 2.0) -> pd.DataFrame:
    """Convenience frame: st_line, trend, rising (line climbing bar over bar)."""
    st, tr = supertrend(high, low, close, period, multiplier)
    rising = st.diff() > 0
    return pd.DataFrame({"st_line": st, "trend": tr, "rising": rising})
