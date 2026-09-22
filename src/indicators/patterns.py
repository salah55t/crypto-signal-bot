"""
Candlestick Pattern Detection
Identifies classic Japanese candlestick patterns that hint at trend
reversals or continuations.

Patterns detected:
  - Bullish: Hammer, Bullish Engulfing, Morning Star, Piercing Line,
            Three White Soldiers, Inverted Hammer, Bullish Harami
  - Bearish: Shooting Star, Bearish Engulfing, Evening Star, Dark Cloud,
             Three Black Crows, Bearish Harami, Hanging Man
  - Neutral: Doji, Spinning Top

Returns a list of detected patterns for the latest candles.
"""
import pandas as pd
import numpy as np


def _candle_body(open_: float, close: float) -> float:
    return abs(close - open_)


def _upper_shadow(open_: float, close: float, high: float) -> float:
    return high - max(open_, close)


def _lower_shadow(open_: float, close: float, low: float) -> float:
    return min(open_, close) - low


def _is_bullish(open_: float, close: float) -> bool:
    return close > open_


def _is_bearish(open_: float, close: float) -> bool:
    return close < open_


def detect_hammer(o: float, h: float, l: float, c: float) -> bool:
    """Hammer: small body at top, long lower shadow (>= 2x body)."""
    body = _candle_body(o, c)
    if body == 0:
        return False
    lower = _lower_shadow(o, c, l)
    upper = _upper_shadow(o, c, h)
    return lower >= 2 * body and upper <= body * 0.3 and body / max(h - l, 1e-9) < 0.4


def detect_shooting_star(o: float, h: float, l: float, c: float) -> bool:
    """Shooting Star: small body at bottom, long upper shadow."""
    body = _candle_body(o, c)
    if body == 0:
        return False
    upper = _upper_shadow(o, c, h)
    lower = _lower_shadow(o, c, l)
    return upper >= 2 * body and lower <= body * 0.3 and body / max(h - l, 1e-9) < 0.4


def detect_doji(o: float, h: float, l: float, c: float,
                threshold: float = 0.1) -> bool:
    """Doji: open and close almost equal."""
    body = _candle_body(o, c)
    range_ = max(h - l, 1e-9)
    return body / range_ < threshold


def detect_engulfing(o1: float, h1: float, l1: float, c1: float,
                     o2: float, h2: float, l2: float, c2: float) -> str:
    """
    Engulfing pattern on two consecutive candles.
    Returns: 'bullish', 'bearish', or 'none'
    """
    body1 = _candle_body(o1, c1)
    body2 = _candle_body(o2, c2)
    if body1 == 0 or body2 == 0:
        return "none"
    if body2 > body1:
        if _is_bearish(o1, c1) and _is_bullish(o2, c2):
            if o2 <= c1 and c2 >= o1:
                return "bullish"
        if _is_bullish(o1, c1) and _is_bearish(o2, c2):
            if o2 >= c1 and c2 <= o1:
                return "bearish"
    return "none"


def detect_harami(o1: float, h1: float, l1: float, c1: float,
                  o2: float, h2: float, l2: float, c2: float) -> str:
    """Harami: small body inside previous large body."""
    body1 = _candle_body(o1, c1)
    body2 = _candle_body(o2, c2)
    if body1 == 0 or body2 == 0:
        return "none"
    if body2 < body1 and max(o2, c2) < max(o1, c1) and min(o2, c2) > min(o1, c1):
        if _is_bearish(o1, c1) and _is_bullish(o2, c2):
            return "bullish"
        if _is_bullish(o1, c1) and _is_bearish(o2, c2):
            return "bearish"
    return "none"


def detect_morning_star(o1, h1, l1, c1, o2, h2, l2, c2,
                        o3, h3, l3, c3) -> bool:
    """Morning Star (3 candles) - bullish reversal at bottom."""
    body1 = _candle_body(o1, c1)
    body2 = _candle_body(o2, c2)
    body3 = _candle_body(o3, c3)
    if body1 == 0 or body3 == 0:
        return False
    return (
        _is_bearish(o1, c1)
        and body2 < body1 * 0.5
        and _is_bullish(o3, c3)
        and c3 > (o1 + c1) / 2
    )


def detect_evening_star(o1, h1, l1, c1, o2, h2, l2, c2,
                         o3, h3, l3, c3) -> bool:
    """Evening Star (3 candles) - bearish reversal at top."""
    body1 = _candle_body(o1, c1)
    body2 = _candle_body(o2, c2)
    body3 = _candle_body(o3, c3)
    if body1 == 0 or body3 == 0:
        return False
    return (
        _is_bullish(o1, c1)
        and body2 < body1 * 0.5
        and _is_bearish(o3, c3)
        and c3 < (o1 + c1) / 2
    )


def detect_piercing_line(o1, h1, l1, c1, o2, h2, l2, c2) -> bool:
    """Piercing Line - bullish reversal."""
    body1 = _candle_body(o1, c1)
    body2 = _candle_body(o2, c2)
    if body1 == 0 or body2 == 0:
        return False
    return (
        _is_bearish(o1, c1)
        and _is_bullish(o2, c2)
        and o2 < l1  # opens below previous low
        and c2 > (o1 + c1) / 2  # closes above midpoint
        and c2 < o1  # but below previous open
    )


def detect_dark_cloud(o1, h1, l1, c1, o2, h2, l2, c2) -> bool:
    """Dark Cloud Cover - bearish reversal."""
    body1 = _candle_body(o1, c1)
    body2 = _candle_body(o2, c2)
    if body1 == 0 or body2 == 0:
        return False
    return (
        _is_bullish(o1, c1)
        and _is_bearish(o2, c2)
        and o2 > h1  # opens above previous high
        and c2 < (o1 + c1) / 2  # closes below midpoint
        and c2 > o1  # but above previous open
    )


def detect_three_white_soldiers(o1, h1, l1, c1, o2, h2, l2, c2,
                                  o3, h3, l3, c3) -> bool:
    """Three White Soldiers - strong bullish."""
    return (
        _is_bullish(o1, c1) and _is_bullish(o2, c2) and _is_bullish(o3, c3)
        and c2 > c1 and c3 > c2  # consecutive higher closes (fixed: was c1 > c1)
        and o2 > o1 and o3 > o2  # consecutive higher opens
    )


def detect_three_black_crows(o1, h1, l1, c1, o2, h2, l2, c2,
                              o3, h3, l3, c3) -> bool:
    """Three Black Crows - strong bearish."""
    return (
        _is_bearish(o1, c1) and _is_bearish(o2, c2) and _is_bearish(o3, c3)
        and c2 < c1 and c3 < c2
        and o2 < o1 and o3 < o2
    )


# ============================================
# ADDITIONAL BULLISH PATTERNS
# ============================================

def detect_inverted_hammer(o: float, h: float, l: float, c: float) -> bool:
    """
    Inverted Hammer: small body at bottom, long upper shadow (>= 2x body).
    Appears after downtrend, signals potential reversal.
    """
    body = _candle_body(o, c)
    if body == 0:
        return False
    upper = _upper_shadow(o, c, h)
    lower = _lower_shadow(o, c, l)
    return (upper >= 2 * body
            and lower <= body * 0.5
            and body / max(h - l, 1e-9) < 0.4
            and c > o  # green
            )


def detect_bullish_marubozu(o: float, h: float, l: float, c: float) -> bool:
    """
    Bullish Marubozu: large bullish body, very small or no shadows.
    Indicates strong buying pressure.
    """
    body = _candle_body(o, c)
    if body == 0:
        return False
    range_ = h - l
    if range_ <= 0:
        return False
    upper = _upper_shadow(o, c, h)
    lower = _lower_shadow(o, c, l)
    return (
        _is_bullish(o, c)
        and body / range_ > 0.85
        and upper < body * 0.1
        and lower < body * 0.1
    )


def detect_tweezer_bottom(o1, h1, l1, c1, o2, h2, l2, c2,
                            tolerance: float = 0.001) -> bool:
    """
    Tweezer Bottom: two consecutive candles with nearly identical lows.
    Appears after downtrend, signals reversal.
    """
    body1 = _candle_body(o1, c1)
    body2 = _candle_body(o2, c2)
    if body1 == 0 or body2 == 0:
        return False
    # First candle bearish, second bullish
    if not (_is_bearish(o1, c1) and _is_bullish(o2, c2)):
        return False
    # Lows are nearly equal (within tolerance)
    low_diff = abs(l1 - l2) / min(l1, l2)
    return low_diff < tolerance


def detect_bullish_tri_star(o1, h1, l1, c1, o2, h2, l2, c2,
                              o3, h3, l3, c3,
                              doji_threshold: float = 0.1) -> bool:
    """
    Bullish Tri-Star: three consecutive Doji candles in a downtrend.
    Very rare, signals major reversal.
    """
    if not (detect_doji(o1, h1, l1, c1, doji_threshold)
            and detect_doji(o2, h2, l2, c2, doji_threshold)
            and detect_doji(o3, h3, l3, c3, doji_threshold)):
        return False
    # Forming a downward pattern: each doji lower than the previous
    return (c2 < c1 and c3 < c2 and l3 < l2 < l1)


def detect_stick_sandwich(o1, h1, l1, c1, o2, h2, l2, c2,
                            o3, h3, l3, c3) -> bool:
    """
    Stick Sandwich: bearish-bullish-bearish with same highs.
    Rare reversal pattern.
    """
    body1 = _candle_body(o1, c1)
    body2 = _candle_body(o2, c2)
    body3 = _candle_body(o3, c3)
    if body1 == 0 or body2 == 0 or body3 == 0:
        return False
    return (
        _is_bearish(o1, c1)
        and _is_bullish(o2, c2)
        and _is_bearish(o3, c3)
        and abs(c1 - c3) / min(c1, c3) < 0.001  # same closing prices
        and c2 > c1 and c2 > c3  # middle candle higher
    )


def detect_abandoned_baby_bottom(o1, h1, l1, c1, o2, h2, l2, c2,
                                    o3, h3, l3, c3) -> bool:
    """
    Abandoned Baby Bottom: very rare bullish reversal.
    Bearish candle, then Doji with gap down, then bullish candle with gap up.
    """
    body1 = _candle_body(o1, c1)
    body3 = _candle_body(o3, c3)
    if body1 == 0 or body3 == 0:
        return False
    return (
        _is_bearish(o1, c1)
        and detect_doji(o2, h2, l2, c2, threshold=0.05)
        and _is_bullish(o3, c3)
        and max(o2, c2) < l1  # gap down (Doji below previous low)
        and min(o2, c2) > h3  # gap up (Doji above next high)
    )


def detect_homing_pigeon(o1, h1, l1, c1, o2, h2, l2, c2) -> bool:
    """
    Homing Pigeon: similar to Bullish Harami but both candles bearish.
    Second smaller candle is "inside" the first larger one.
    """
    body1 = _candle_body(o1, c1)
    body2 = _candle_body(o2, c2)
    if body1 == 0 or body2 == 0:
        return False
    return (
        _is_bearish(o1, c1)
        and _is_bearish(o2, c2)
        and body2 < body1 * 0.6  # smaller body
        and max(o2, c2) < max(o1, c1)
        and min(o2, c2) > min(o1, c1)
    )


def detect_all_patterns(df: pd.DataFrame, lookback: int = 3) -> list:
    """
    Run all pattern detectors on the latest candles.
    Returns a list of {pattern, direction, signal_strength}.
    """
    if len(df) < 3:
        return []
    last = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]
    o1, h1, l1, c1 = prev2["open"], prev2["high"], prev2["low"], prev2["close"]
    o2, h2, l2, c2 = prev["open"], prev["high"], prev["low"], prev["close"]
    o3, h3, l3, c3 = last["open"], last["high"], last["low"], last["close"]

    patterns = []

    # Single-candle patterns
    if detect_hammer(o3, h3, l3, c3):
        patterns.append({"pattern": "Hammer", "direction": "bullish", "strength": 0.6})
    if detect_shooting_star(o3, h3, l3, c3):
        patterns.append({"pattern": "Shooting Star", "direction": "bearish", "strength": 0.6})
    if detect_doji(o3, h3, l3, c3):
        patterns.append({"pattern": "Doji", "direction": "neutral", "strength": 0.3})

    # Two-candle patterns
    eng = detect_engulfing(o2, h2, l2, c2, o3, h3, l3, c3)
    if eng != "none":
        patterns.append({
            "pattern": f"{eng.capitalize()} Engulfing",
            "direction": eng,
            "strength": 0.8 if eng == "bullish" else 0.8,
        })
    har = detect_harami(o2, h2, l2, c2, o3, h3, l3, c3)
    if har != "none":
        patterns.append({
            "pattern": f"{har.capitalize()} Harami",
            "direction": har,
            "strength": 0.5,
        })
    if detect_piercing_line(o2, h2, l2, c2, o3, h3, l3, c3):
        patterns.append({"pattern": "Piercing Line", "direction": "bullish", "strength": 0.7})
    if detect_dark_cloud(o2, h2, l2, c2, o3, h3, l3, c3):
        patterns.append({"pattern": "Dark Cloud Cover", "direction": "bearish", "strength": 0.7})

    # Three-candle patterns
    if detect_morning_star(o1, h1, l1, c1, o2, h2, l2, c2, o3, h3, l3, c3):
        patterns.append({"pattern": "Morning Star", "direction": "bullish", "strength": 0.9})
    if detect_evening_star(o1, h1, l1, c1, o2, h2, l2, c2, o3, h3, l3, c3):
        patterns.append({"pattern": "Evening Star", "direction": "bearish", "strength": 0.9})
    if detect_three_white_soldiers(o1, h1, l1, c1, o2, h2, l2, c2, o3, h3, l3, c3):
        patterns.append({"pattern": "Three White Soldiers", "direction": "bullish", "strength": 0.9})
    if detect_three_black_crows(o1, h1, l1, c1, o2, h2, l2, c2, o3, h3, l3, c3):
        patterns.append({"pattern": "Three Black Crows", "direction": "bearish", "strength": 0.9})

    # === NEW ADDITIONAL BULLISH PATTERNS ===
    if detect_inverted_hammer(o3, h3, l3, c3):
        patterns.append({"pattern": "Inverted Hammer", "direction": "bullish", "strength": 0.7})
    if detect_bullish_marubozu(o3, h3, l3, c3):
        patterns.append({"pattern": "Bullish Marubozu", "direction": "bullish", "strength": 0.85})
    if detect_tweezer_bottom(o2, h2, l2, c2, o3, h3, l3, c3):
        patterns.append({"pattern": "Tweezer Bottom", "direction": "bullish", "strength": 0.8})
    if detect_bullish_tri_star(o1, h1, l1, c1, o2, h2, l2, c2, o3, h3, l3, c3):
        patterns.append({"pattern": "Bullish Tri-Star", "direction": "bullish", "strength": 0.95})
    if detect_stick_sandwich(o1, h1, l1, c1, o2, h2, l2, c2, o3, h3, l3, c3):
        patterns.append({"pattern": "Stick Sandwich", "direction": "bullish", "strength": 0.85})
    if detect_abandoned_baby_bottom(o1, h1, l1, c1, o2, h2, l2, c2, o3, h3, l3, c3):
        patterns.append({"pattern": "Abandoned Baby Bottom", "direction": "bullish", "strength": 0.95})
    if detect_homing_pigeon(o2, h2, l2, c2, o3, h3, l3, c3):
        patterns.append({"pattern": "Homing Pigeon", "direction": "bullish", "strength": 0.75})

    return patterns
