"""
Volume Analysis Module
Computes volume-based metrics for signal confirmation.

Components:
  - Volume spike detection (current vs avg)
  - Cumulative Volume Delta (proxy from taker buy/sell)
  - Volume Profile (price-volume distribution)
  - Volume Trend (volume slope, increasing/decreasing)
"""
import pandas as pd
import numpy as np


def volume_spike(volume: pd.Series, period: int = 20,
                 threshold: float = 1.5) -> pd.Series:
    """Boolean series: True if current volume > threshold * SMA(volume)."""
    sma = volume.rolling(window=period, min_periods=period).mean()
    return (volume / sma.replace(0, np.nan)) >= threshold


def cumulative_volume_delta(df: pd.DataFrame) -> pd.Series:
    """
    Approximate CVD using taker buy/sell volumes.
    Positive = buyers aggressive, Negative = sellers aggressive.
    """
    if "taker_buy_base" not in df.columns:
        return pd.Series(0, index=df.index)
    buy_vol = df["taker_buy_base"]
    sell_vol = df["volume"] - df["taker_buy_base"]
    delta = buy_vol - sell_vol
    return delta.cumsum()


def volume_trend_slope(volume: pd.Series, period: int = 20) -> float:
    """Returns slope of volume regression (positive = rising)."""
    v = volume.iloc[-period:].values
    if len(v) < period or np.std(v) == 0:
        return 0.0
    x = np.arange(period, dtype=float)
    y = v
    slope = np.polyfit(x, y, 1)[0]
    # Normalize by mean volume for scale invariance
    return float(slope / max(np.mean(v), 1e-9))


def volume_profile(df: pd.DataFrame, bins: int = 20) -> pd.DataFrame:
    """
    Build a simple Volume Profile: distribute volume across price bins.
    Returns DataFrame with columns: price_low, price_high, volume, pct.
    """
    if len(df) < bins:
        return pd.DataFrame()
    # Use close price for binning
    prices = df["close"].values
    volumes = df["volume"].values
    price_min = prices.min()
    price_max = prices.max()
    if price_max == price_min:
        return pd.DataFrame()
    edges = np.linspace(price_min, price_max, bins + 1)
    bin_idx = np.clip(
        np.digitize(prices, edges) - 1, 0, bins - 1
    )
    bin_volumes = np.zeros(bins)
    np.add.at(bin_volumes, bin_idx, volumes)
    pct = bin_volumes / bin_volumes.sum()
    return pd.DataFrame({
        "price_low": edges[:-1],
        "price_high": edges[1:],
        "volume": bin_volumes,
        "pct": pct,
    })


def find_poc(volume_profile_df: pd.DataFrame) -> float:
    """Point of Control: price level with highest volume."""
    if volume_profile_df.empty:
        return np.nan
    row = volume_profile_df.loc[volume_profile_df["volume"].idxmax()]
    return (row["price_low"] + row["price_high"]) / 2


def volume_at_price_strength(df: pd.DataFrame, current_price: float,
                              bins: int = 20) -> float:
    """Returns volume fraction near current price (within 1 bin)."""
    vp = volume_profile(df, bins=bins)
    if vp.empty:
        return 0.0
    diffs = (vp["price_low"] + vp["price_high"]) / 2 - current_price
    bin_width = vp["price_high"].iloc[0] - vp["price_low"].iloc[0]
    near = (diffs.abs() <= bin_width)
    return float(vp.loc[near, "pct"].sum())


def analyze_volume(df: pd.DataFrame, period: int = 20) -> dict:
    """One-shot volume summary for the latest bar."""
    if len(df) < period + 1:
        return {}
    vol = df["volume"]
    current_vol = vol.iloc[-1]
    avg_vol = vol.iloc[-period:].mean()
    spike_ratio = current_vol / max(avg_vol, 1e-9)
    cvd_now = cumulative_volume_delta(df).iloc[-1]
    cvd_prev = cumulative_volume_delta(df).iloc[-2] if len(df) > 1 else 0
    cvd_delta = cvd_now - cvd_prev
    slope = volume_trend_slope(vol, period)
    vp = volume_profile(df, bins=20)
    poc = find_poc(vp)
    return {
        "current_volume": float(current_vol),
        "avg_volume": float(avg_vol),
        "spike_ratio": float(spike_ratio),
        "is_spike": bool(spike_ratio >= 1.5),
        "cvd": float(cvd_now),
        "cvd_delta": float(cvd_delta),
        "cvd_trend": "up" if cvd_delta > 0 else "down",
        "volume_slope": slope,
        "volume_trend": "rising" if slope > 0.01 else (
            "falling" if slope < -0.01 else "flat"
        ),
        "poc": float(poc) if not np.isnan(poc) else None,
        "volume_profile": vp.to_dict("records") if not vp.empty else [],
    }
