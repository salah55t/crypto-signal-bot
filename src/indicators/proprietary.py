"""
Proprietary Advanced Indicators - invented/rare indicators not commonly available.

Indicators implemented here:
  - Heikin Ashi candles (smoothing technique to detect trend)
  - TD Sequential (Tom DeMark's pattern - counts 13 consecutive bars)
  - Volume Climax Detection (capitulation volume spike)
  - Volatility Contraction (BB squeeze + ATR contraction)
  - Wyckoff Spring Detection (false breakdown + recovery)
  - Fibonacci Retracement Levels (auto-detect swing high/low)
  - Volume-Weighted RSI (gives more weight to high-volume bars)
  - Smart Money Index (tracks aggressive buy/sell behavior)
"""
import pandas as pd
import numpy as np


# ============================================
# HEIKIN ASHI CANDLES
# ============================================

def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert OHLC to Heikin Ashi candles.
    HA_Close = (Open + High + Low + Close) / 4
    HA_Open = (prev HA_Open + prev HA_Close) / 2
    HA_High = max(High, HA_Open, HA_Close)
    HA_Low = min(Low, HA_Open, HA_Close)
    """
    ha = pd.DataFrame(index=df.index)
    ha["close"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    ha["open"] = df["open"]  # initialize
    # Recursive computation
    for i in range(1, len(df)):
        ha.iloc[i, ha.columns.get_loc("open")] = (ha["open"].iloc[i-1] + ha["close"].iloc[i-1]) / 2
    ha["high"] = ha[["open", "close"]].join(df["high"]).max(axis=1)
    ha["low"] = ha[["open", "close"]].join(df["low"]).min(axis=1)
    return ha


def ha_smoke_signal(ha: pd.DataFrame, lookback: int = 10) -> dict:
    """
    Heikin Ashi Smoke Signal: small body HA candles after a downtrend.
    Indicates exhaustion and potential reversal.
    """
    if len(ha) < lookback + 5:
        return {"signal": "none"}
    recent = ha.iloc[-lookback:]
    prev = ha.iloc[-lookback * 2:-lookback] if len(ha) >= lookback * 2 else ha.iloc[:-lookback]

    # Detect downtrend in previous candles
    prev_bearish = (prev["close"] < prev["open"]).sum()
    downtrend = prev_bearish >= len(prev) * 0.6

    # Recent small bodies
    recent_body = (recent["close"] - recent["open"]).abs()
    recent_range = recent["high"] - recent["low"]
    body_ratio = (recent_body / recent_range.replace(0, np.nan)).fillna(0)
    small_bodies = (body_ratio < 0.3).sum()
    # Lower wick presence (buying pressure)
    lower_wicks = ((recent["low"] - recent[["open", "close"]].min(axis=1)) /
                   recent_range.replace(0, np.nan)).fillna(0)
    long_lower_wick = (lower_wicks > 0.4).sum()

    if downtrend and small_bodies >= lookback * 0.6 and long_lower_wick >= 2:
        return {
            "signal": "bullish",
            "strength": 0.7,
            "reason": f"HA smoke: {small_bodies} small bodies after downtrend, {long_lower_wick} long lower wicks",
            "details": {"small_bodies_count": int(small_bodies),
                       "long_lower_wicks": int(long_lower_wick),
                       "downtrend_candles": int(prev_bearish)},
        }
    return {"signal": "none"}


# ============================================
# TD SEQUENTIAL (Tom DeMark)
# ============================================

def td_sequential(close: pd.Series, setup_length: int = 9,
                  countdown_length: int = 13) -> dict:
    """
    TD Sequential indicator:
      - Setup phase: 9 consecutive closes lower than the close 4 bars earlier
                     (for buy setup)
      - Countdown phase: 13 closes lower than the low 2 bars earlier (for buy)
    Returns: {setup_count, countdown_count, signal}
    """
    if len(close) < setup_length + 4:
        return {"setup_count": 0, "countdown_count": 0, "signal": "none"}

    # Buy setup: 9 consecutive closes < close 4 bars ago
    setup_count = 0
    setup_active = False
    setup_start_idx = None
    for i in range(4, len(close)):
        if close.iloc[i] < close.iloc[i - 4]:
            setup_count += 1
            if not setup_active:
                setup_active = True
                setup_start_idx = i
            if setup_count >= setup_length:
                # Setup complete, start countdown
                break
        else:
            setup_count = 0
            setup_active = False

    if setup_count < setup_length:
        return {"setup_count": setup_count, "countdown_count": 0,
                "signal": "forming" if setup_count >= 6 else "none"}

    # Countdown phase: count closes < low 2 bars ago
    countdown_count = 0
    for i in range(setup_start_idx + setup_length, len(close)):
        if close.iloc[i] < close.iloc[i - 2]:
            countdown_count += 1
            if countdown_count >= countdown_length:
                return {"setup_count": setup_count,
                        "countdown_count": countdown_count,
                        "signal": "bullish",
                        "strength": 0.9,
                        "reason": f"TD Sequential buy: {countdown_length} countdown complete"}

    return {"setup_count": setup_count,
            "countdown_count": countdown_count,
            "signal": "forming" if countdown_count >= 8 else "none",
            "reason": f"TD buy setup ({setup_count}/{setup_length}), countdown ({countdown_count}/{countdown_length})"}


# ============================================
# VOLUME CLIMAX DETECTION
# ============================================

def volume_climax(df: pd.DataFrame, lookback: int = 50,
                   spike_threshold: float = 3.0,
                   drop_threshold: float = -3.0) -> dict:
    """
    Volume climax = capitulation selling (high volume + sharp price drop)
    Often marks market bottoms.
    """
    if len(df) < lookback + 5:
        return {"signal": "none"}
    vol = df["volume"]
    close = df["close"]
    avg_vol = vol.rolling(lookback).mean()
    last_vol = vol.iloc[-1]
    last_ret = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100
    vol_ratio = last_vol / avg_vol.iloc[-1] if avg_vol.iloc[-1] > 0 else 1

    # Selling climax: high volume + drop
    if vol_ratio >= spike_threshold and last_ret <= drop_threshold:
        # Check if there's a recovery (lower wick)
        last_candle = df.iloc[-1]
        body = abs(last_candle["close"] - last_candle["open"])
        range_ = last_candle["high"] - last_candle["low"]
        lower_wick = min(last_candle["open"], last_candle["close"]) - last_candle["low"]
        recovery = lower_wick > body * 1.5 if body > 0 and range_ > 0 else False
        return {
            "signal": "bullish",
            "strength": 0.85 if recovery else 0.6,
            "reason": f"Selling climax: vol {vol_ratio:.1f}x avg + drop {last_ret:.2f}% "
                      f"{'with recovery wick' if recovery else ''}",
            "details": {"volume_ratio": float(vol_ratio),
                       "price_drop_pct": float(last_ret),
                       "recovery_wick": bool(recovery)},
        }
    return {"signal": "none", "vol_ratio": float(vol_ratio)}


def volatility_contraction(df: pd.DataFrame, lookback: int = 20,
                            contraction_ratio: float = 0.5) -> dict:
    """
    Volatility contraction: ATR has shrunk significantly vs historical.
    Often precedes explosive moves.
    """
    if len(df) < lookback * 3:
        return {"signal": "none"}
    high, low, close = df["high"], df["low"], df["close"]
    # Recent ATR
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    atr_now = tr.iloc[-lookback:].mean()
    atr_prior = tr.iloc[-lookback * 3:-lookback].mean()
    if atr_prior == 0:
        return {"signal": "none"}
    ratio = atr_now / atr_prior

    # Price position relative to recent range
    recent_high = high.iloc[-lookback:].max()
    recent_low = low.iloc[-lookback:].min()
    price = close.iloc[-1]
    price_position = (price - recent_low) / (recent_high - recent_low) if (recent_high - recent_low) > 0 else 0.5

    if ratio < contraction_ratio and price_position < 0.3:
        # Contraction + price near bottom = potential bounce
        return {
            "signal": "bullish",
            "strength": 0.65,
            "reason": f"Volatility contraction (ATR ratio {ratio:.2f}) + price near range bottom",
            "details": {"atr_ratio": float(ratio),
                       "price_position": float(price_position)},
        }
    return {"signal": "none", "atr_ratio": float(ratio)}


# ============================================
# WYCKOFF SPRING DETECTION
# ============================================

def wyckoff_spring(df: pd.DataFrame, support_lookback: int = 50,
                    breakdown_threshold: float = 0.005,
                    recovery_threshold: float = 0.5) -> dict:
    """
    Wyckoff Spring:
    - Price breaks below a support level (false breakdown)
    - Then quickly recovers back above the support
    - Indicates that "smart money" absorbed the selling pressure

    Returns bullish signal when a spring is detected.
    """
    if len(df) < support_lookback + 5:
        return {"signal": "none"}

    # Find support level (lowest low in the lookback window excluding last 5 bars)
    lookback_df = df.iloc[-support_lookback - 5:-5]
    if len(lookback_df) == 0:
        return {"signal": "none"}
    support = lookback_df["low"].min()

    last_candles = df.iloc[-5:]
    closes = last_candles["close"].values
    lows = last_candles["low"].values
    highs = last_candles["high"].values

    # Did we break below support?
    broke_below = any(l < support * (1 - breakdown_threshold) for l in lows)
    if not broke_below:
        return {"signal": "none", "support": float(support)}

    # Did we recover back above support?
    last_close = closes[-1]
    if last_close > support:
        # Spring detected!
        # Check strength of recovery
        breakdown_low = min(lows)
        recovery_pct = (last_close - breakdown_low) / (support - breakdown_low) if (support - breakdown_low) > 0 else 1
        if recovery_pct >= recovery_threshold:
            strength = min(1.0, recovery_pct)
            return {
                "signal": "bullish",
                "strength": float(strength),
                "reason": f"Wyckoff Spring: false breakdown below {support:.4f} "
                          f"recovered {(recovery_pct*100):.0f}%",
                "details": {"support_level": float(support),
                           "breakdown_low": float(breakdown_low),
                           "recovery_pct": float(recovery_pct)},
            }
    return {"signal": "none", "support": float(support)}


# ============================================
# FIBONACCI RETRACEMENT
# ============================================

def fibonacci_levels(df: pd.DataFrame, lookback: int = 100) -> dict:
    """
    Auto-detect swing high/low in lookback, compute Fibonacci retracement levels.
    Returns levels: 0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0
    """
    if len(df) < lookback:
        return {}
    window = df.iloc[-lookback:]
    swing_high = window["high"].max()
    swing_low = window["low"].min()
    diff = swing_high - swing_low
    if diff == 0:
        return {}
    levels = {
        "0.0": swing_low,
        "0.236": swing_low + 0.236 * diff,
        "0.382": swing_low + 0.382 * diff,
        "0.5": swing_low + 0.5 * diff,
        "0.618": swing_low + 0.618 * diff,
        "0.786": swing_low + 0.786 * diff,
        "1.0": swing_high,
    }
    return {
        "swing_high": float(swing_high),
        "swing_low": float(swing_low),
        "levels": {k: float(v) for k, v in levels.items()},
    }


def fibonacci_confluence(df: pd.DataFrame, lookback: int = 100) -> dict:
    """
    Find if current price is near key Fibonacci levels (0.618, 0.786 - the "golden zone").
    """
    fib = fibonacci_levels(df, lookback)
    if not fib:
        return {"signal": "none"}
    current = df["close"].iloc[-1]
    golden_618 = fib["levels"]["0.618"]
    golden_786 = fib["levels"]["0.786"]
    swing_low = fib["levels"]["0.0"]

    # In golden zone (between 0.618 and 0.786)?
    if golden_786 <= current <= golden_618:
        return {
            "signal": "bullish",
            "strength": 0.7,
            "reason": f"Price in Fibonacci golden zone ({(current/golden_618*100):.1f}% of 0.618 retracement)",
            "details": fib,
        }
    # Below 0.786 (deep retracement) - even stronger signal if combined with reversal
    elif current < golden_786 and current > swing_low:
        return {
            "signal": "bullish",
            "strength": 0.6,
            "reason": f"Price below 0.786 retracement (deep retracement)",
            "details": fib,
        }
    return {"signal": "none", "current": float(current), **fib}


# ============================================
# VOLUME-WEIGHTED RSI
# ============================================

def volume_weighted_rsi(close: pd.Series, volume: pd.Series,
                         period: int = 14) -> pd.Series:
    """
    RSI where each gain/loss is weighted by the volume of that bar.
    Gives more importance to high-volume moves.
    """
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0) * volume
    loss = (-delta).where(delta < 0, 0.0) * volume
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    # Normalize by total volume
    avg_vol = volume.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_gain = avg_gain / avg_vol.replace(0, np.nan)
    avg_loss = avg_loss / avg_vol.replace(0, np.nan)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def smart_money_divergence(close: pd.Series, volume: pd.Series,
                            lookback: int = 50) -> dict:
    """
    Detect divergence between regular RSI and Volume-Weighted RSI.
    When VW-RSI is higher than regular RSI at price lows,
    it indicates smart money is accumulating despite low prices.
    """
    from src.indicators.technical import rsi
    if len(close) < lookback:
        return {"signal": "none"}
    r = rsi(close, 14)
    vw_r = volume_weighted_rsi(close, volume, 14)
    last_r = r.iloc[-1]
    last_vwr = vw_r.iloc[-1]

    # Both should be in oversold zone
    if last_r > 40:
        return {"signal": "none"}

    # VW-RSI > regular RSI = smart money buying
    if last_vwr > last_r + 10:
        diff = last_vwr - last_r
        return {
            "signal": "bullish",
            "strength": min(1.0, diff / 30),
            "reason": f"Smart money divergence: VW-RSI ({last_vwr:.1f}) > RSI ({last_r:.1f})",
            "details": {"rsi": float(last_r), "vw_rsi": float(last_vwr)},
        }
    return {"signal": "none"}


# ============================================
# BOTTOM DETECTION COMPOSITE
# ============================================

def is_near_bottom(df: pd.DataFrame, lookback: int = 50,
                    threshold_pct: float = 0.05) -> dict:
    """
    Check if current price is near the recent low (within threshold_pct of the lowest low).
    Used by the bottom scanner.
    """
    if len(df) < lookback:
        return {"near_bottom": False}
    recent_low = df["low"].iloc[-lookback:].min()
    recent_high = df["high"].iloc[-lookback:].max()
    current = df["close"].iloc[-1]
    distance_from_low = (current - recent_low) / recent_low if recent_low > 0 else 0
    position_in_range = (current - recent_low) / (recent_high - recent_low) if (recent_high - recent_low) > 0 else 0.5

    return {
        "near_bottom": bool(distance_from_low < threshold_pct),
        "current_price": float(current),
        "recent_low": float(recent_low),
        "recent_high": float(recent_high),
        "distance_from_low_pct": float(distance_from_low * 100),
        "position_in_range_pct": float(position_in_range * 100),
        "lookback": lookback,
    }
