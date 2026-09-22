"""
Technical Indicators Module
Computes a comprehensive suite of technical indicators on OHLCV data.

Indicators:
  - Trend: SMA, EMA, WMA, VWAP, MACD, ADX, Parabolic SAR, Ichimoku Cloud
  - Momentum: RSI, Stochastic, Williams %R, CCI, ROC, Momentum
  - Volatility: Bollinger Bands, ATR, Keltner Channels, Standard Deviation
  - Volume: OBV, MFI, CMF, Volume SMA, VWAP

Each function returns a pandas Series / DataFrame with named columns.
"""
import pandas as pd
import numpy as np


# ============================================
# TREND INDICATORS
# ============================================

def sma(series: pd.Series, period: int = 20) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int = 20) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def wma(series: pd.Series, period: int = 20) -> pd.Series:
    """Weighted Moving Average."""
    weights = np.arange(1, period + 1, dtype=float)
    weights = weights / weights.sum()
    return series.rolling(window=period).apply(
        lambda x: np.dot(x, weights) if len(x) == period else np.nan, raw=True
    )


def macd(series: pd.Series, fast: int = 12, slow: int = 26,
         signal: int = 9) -> pd.DataFrame:
    """MACD: line, signal, histogram."""
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return pd.DataFrame({
        "macd": macd_line,
        "signal": signal_line,
        "histogram": hist,
    })


def adx(high: pd.Series, low: pd.Series, close: pd.Series,
        period: int = 14) -> pd.DataFrame:
    """Average Directional Index (ADX) with +DI and -DI.

    Implements Wilder's original algorithm correctly:
      +DM = up_move  when up_move > down_move AND up_move > 0, else 0
      -DM = down_move when down_move > up_move AND down_move > 0, else 0
    where up_move = high - prev_high, down_move = prev_low - low.
    Both DM and TR are smoothed with Wilder's smoothing (EWM alpha=1/period).
    """
    high = high.astype(float)
    low = low.astype(float)
    close = close.astype(float)

    up_move = high.diff()
    down_move = -low.diff()  # prev_low - low

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

    alpha = 1 / period
    atr = tr.ewm(alpha=alpha, adjust=False, min_periods=period).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=alpha, adjust=False, min_periods=period).mean() / atr)
    di_sum = (plus_di + minus_di).replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    adx_val = dx.ewm(alpha=alpha, adjust=False, min_periods=period).mean()

    return pd.DataFrame({
        "adx": adx_val,
        "plus_di": plus_di,
        "minus_di": minus_di,
    })


def vwap(high: pd.Series, low: pd.Series, close: pd.Series,
         volume: pd.Series) -> pd.Series:
    """Volume Weighted Average Price."""
    typical_price = (high + low + close) / 3
    return (typical_price * volume).cumsum() / volume.cumsum()


def vwap_rolling(high: pd.Series, low: pd.Series, close: pd.Series,
                 volume: pd.Series, window: int = 20) -> pd.Series:
    """Rolling VWAP (anchored to window)."""
    typical_price = (high + low + close) / 3
    cum_vol_price = (typical_price * volume).rolling(window=window, min_periods=1).sum()
    cum_vol = volume.rolling(window=window, min_periods=1).sum()
    return cum_vol_price / cum_vol


# ============================================
# MOMENTUM INDICATORS
# ============================================

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index (Wilder's smoothing)."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
               k_period: int = 14, d_period: int = 3) -> pd.DataFrame:
    """Stochastic Oscillator (%K, %D)."""
    lowest_low = low.rolling(window=k_period, min_periods=k_period).min()
    highest_high = high.rolling(window=k_period, min_periods=k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    d = k.rolling(window=d_period, min_periods=d_period).mean()
    return pd.DataFrame({"k": k, "d": d})


def williams_r(high: pd.Series, low: pd.Series, close: pd.Series,
               period: int = 14) -> pd.Series:
    """Williams %R."""
    highest_high = high.rolling(window=period, min_periods=period).max()
    lowest_low = low.rolling(window=period, min_periods=period).min()
    return -100 * (highest_high - close) / (highest_high - lowest_low).replace(0, np.nan)


def cci(high: pd.Series, low: pd.Series, close: pd.Series,
        period: int = 20) -> pd.Series:
    """Commodity Channel Index."""
    tp = (high + low + close) / 3
    sma_tp = tp.rolling(window=period, min_periods=period).mean()
    mean_dev = tp.rolling(window=period, min_periods=period).apply(
        lambda x: np.abs(x - x.mean()).mean() if len(x) == period else np.nan, raw=True
    )
    return (tp - sma_tp) / (0.015 * mean_dev.replace(0, np.nan))


def roc(series: pd.Series, period: int = 12) -> pd.Series:
    """Rate of Change."""
    return (series - series.shift(period)) / series.shift(period) * 100


def momentum(series: pd.Series, period: int = 10) -> pd.Series:
    """Momentum (price - price[n periods ago])."""
    return series - series.shift(period)


# ============================================
# VOLATILITY INDICATORS
# ============================================

def bollinger_bands(series: pd.Series, period: int = 20,
                    std_dev: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands: middle (SMA), upper, lower."""
    sma = series.rolling(window=period, min_periods=period).mean()
    std = series.rolling(window=period, min_periods=period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    bandwidth = (upper - lower) / sma
    percent_b = (series - lower) / (upper - lower).replace(0, np.nan)
    return pd.DataFrame({
        "middle": sma,
        "upper": upper,
        "lower": lower,
        "bandwidth": bandwidth,
        "percent_b": percent_b,
    })


def atr(high: pd.Series, low: pd.Series, close: pd.Series,
        period: int = 14) -> pd.Series:
    """Average True Range."""
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()


def keltner_channels(high: pd.Series, low: pd.Series, close: pd.Series,
                     period: int = 20, atr_mult: float = 1.5) -> pd.DataFrame:
    """Keltner Channels: EMA middle, ATR-based bands."""
    middle = ema(close, period)
    atr_val = atr(high, low, close, period)
    return pd.DataFrame({
        "middle": middle,
        "upper": middle + atr_mult * atr_val,
        "lower": middle - atr_mult * atr_val,
    })


# ============================================
# VOLUME INDICATORS
# ============================================

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume."""
    direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    return (direction * volume).cumsum()


def mfi(high: pd.Series, low: pd.Series, close: pd.Series,
        volume: pd.Series, period: int = 14) -> pd.Series:
    """Money Flow Index."""
    tp = (high + low + close) / 3
    mf = tp * volume
    delta = tp.diff()
    positive_mf = mf.where(delta > 0, 0.0).rolling(window=period, min_periods=period).sum()
    negative_mf = mf.where(delta < 0, 0.0).rolling(window=period, min_periods=period).sum()
    mfr = positive_mf / negative_mf.replace(0, np.nan)
    return 100 - (100 / (1 + mfr))


def cmf(high: pd.Series, low: pd.Series, close: pd.Series,
        volume: pd.Series, period: int = 20) -> pd.Series:
    """Chaikin Money Flow."""
    mfv = ((close - low) - (high - close)) / (high - low).replace(0, np.nan) * volume
    return mfv.rolling(window=period, min_periods=period).sum() / \
           volume.rolling(window=period, min_periods=period).sum()


def volume_sma(volume: pd.Series, period: int = 20) -> pd.Series:
    """Volume Simple Moving Average."""
    return volume.rolling(window=period, min_periods=period).mean()


def volume_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
    """Current volume / SMA(volume)."""
    return volume / volume_sma(volume, period).replace(0, np.nan)


# ============================================
# TREND DETECTION HELPERS
# ============================================

def detect_trend(close: pd.Series, short_period: int = 20,
                 long_period: int = 50) -> pd.Series:
    """
    Trend classification based on EMA cross.
    1 = uptrend, -1 = downtrend, 0 = sideways
    """
    ema_short = ema(close, short_period)
    ema_long = ema(close, long_period)
    diff = (ema_short - ema_long) / ema_long.replace(0, np.nan)
    return pd.Series(
        np.where(diff > 0.005, 1, np.where(diff < -0.005, -1, 0)),
        index=close.index
    )


def detect_divergence(price: pd.Series, oscillator: pd.Series,
                      lookback: int = 50) -> str:
    """
    Simple divergence detection:
      - Bullish: price makes lower low, oscillator makes higher low
      - Bearish: price makes higher high, oscillator makes lower high
    Returns: 'bullish', 'bearish', or 'none'
    """
    if len(price) < lookback or len(oscillator) < lookback:
        return "none"
    p = price.iloc[-lookback:]
    o = oscillator.iloc[-lookback:]
    # Find local extremes (simple argmin/argmax in halves)
    half = lookback // 2
    p_recent_low = p.iloc[-half:].min()
    p_prior_low = p.iloc[:-half].min()
    p_recent_high = p.iloc[-half:].max()
    p_prior_high = p.iloc[:-half].max()
    o_recent_low = o.iloc[-half:].min()
    o_prior_low = o.iloc[:-half].min()
    o_recent_high = o.iloc[-half:].max()
    o_prior_high = o.iloc[:-half].max()

    if p_recent_low < p_prior_low and o_recent_low > o_prior_low:
        return "bullish"
    if p_recent_high > p_prior_high and o_recent_high < o_prior_high:
        return "bearish"
    return "none"
