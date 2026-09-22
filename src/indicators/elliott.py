"""
Elliott Wave Engine (v4) — Market Cycle Layer

Answers "WHERE are we in the cycle?" — the timing layer of the bot.

Classical Elliott rules encoded (pragmatic, tolerance-aware):
  - Wave 2 never retraces more than 100% of wave 1
  - Wave 3 is never the shortest impulse wave (5% tolerance)
  - Wave 4 does not enter wave 1 price territory (0.3% tolerance)
  - Fibonacci relationships boost confidence:
      wave 2 ~ 0.5-0.618 of wave 1 (golden retrace)
      wave 3 ~ 1.618 x wave 1
      wave 4 ~ 0.382 of wave 3
      wave 5 ~ 0.618 x wave 3 (used as the projection target)
  - ABC corrections: C ~ 1.0-1.618 x A, B stays below the impulse origin

Detected patterns (mirror logic for bearish):
  impulse_up           full 5-wave up impulse (complete)
  impulse_down         full 5-wave down impulse (complete)
  partial_impulse_up   wave 4 confirmed, wave 4/5 zone
  partial_impulse_down
  wave3_up / wave3_down  wave 3 in progress (strongest phase)
  correction_after_up    ABC down after an up impulse
  correction_after_down  ABC up after a down impulse
  unclear              no readable structure (no adjustment applied)

Trade implications (extracted by the confluence engine):
  wave 3 in progress   -> strongest continuation phase (boost)
  wave 4 pullback      -> good R/R entry, wave-5 projection target (boost)
  wave 5 mature        -> final leg, correction risk (penalty when mature)
  impulse complete     -> correction expected (penalty for continuation)
  C complete           -> new cycle starting (reversal boost)
"""
import pandas as pd
from typing import Dict, List, Optional, Tuple

from src.indicators.fibonacci import find_swing_points

# Tolerances
OVERLAP_TOLERANCE_PCT = 0.003   # wave 4 may pierce wave 1 territory by 0.3%
WAVE3_SHORT_TOLERANCE = 0.95    # wave 3 may be up to 5% shorter than min(w1,w5)

# Fibonacci windows used for confidence scoring
FIB_IDEAL = {"w2": (0.5, 0.618), "w3_over_w1": (1.5, 1.75),
             "w4": (0.3, 0.5), "w5_over_w3": (0.5, 1.0),
             "c_over_a": (0.8, 1.3)}

WAVE5_PROJ_RATIO = 0.618  # projected wave 5 length = 0.618 x wave 3


def _zigzag(df: pd.DataFrame, lookback: int = 150,
            strength: int = 3) -> List[Tuple[int, float, str]]:
    """
    Build an alternating zigzag pivot sequence from fractal swing points.
    Returns [(index, price, "H"|"L"), ...] oldest first, strictly alternating.
    """
    window = df.iloc[-lookback:] if len(df) > lookback else df
    highs, lows = find_swing_points(window, lookback=len(window),
                                    strength=strength)
    events = sorted(
        [(i, p, "H") for i, p in highs] + [(i, p, "L") for i, p in lows]
    )
    zz: List[Tuple[int, float, str]] = []
    for i, p, typ in events:
        if zz and zz[-1][2] == typ:
            # Same type twice: keep the more extreme pivot
            if (typ == "H" and p >= zz[-1][1]) or \
               (typ == "L" and p <= zz[-1][1]):
                zz[-1] = (i, p, typ)
        else:
            zz.append((i, p, typ))
    return zz


def _fib_hit(value: float, lo: float, hi: float) -> bool:
    return lo <= value <= hi


def _impulse_confidence(w2: float, w3_w1: float, w4: float,
                        w5_w3: Optional[float] = None,
                        w5_w1: Optional[float] = None,
                        base: float = 0.4) -> Tuple[float, List[str]]:
    """Score a fitted impulse against classic Fibonacci relationships."""
    hits: List[str] = []
    conf = base
    if _fib_hit(w2, *FIB_IDEAL["w2"]):
        conf += 0.30
        hits.append("wave2 golden retrace (0.5-0.618)")
    elif _fib_hit(w2, 0.382, 0.786):
        conf += 0.20
        hits.append("wave2 fib retrace (0.382-0.786)")
    if _fib_hit(w3_w1, *FIB_IDEAL["w3_over_w1"]):
        conf += 0.20
        hits.append("wave3 ~1.618 x wave1")
    elif 1.2 <= w3_w1 <= 2.2:
        conf += 0.10
        hits.append("wave3 extended vs wave1")
    if _fib_hit(w4, *FIB_IDEAL["w4"]):
        conf += 0.20
        hits.append("wave4 shallow retrace (0.3-0.5)")
    elif _fib_hit(w4, 0.236, 0.618):
        conf += 0.10
        hits.append("wave4 fib retrace")
    if w5_w3 is not None and _fib_hit(w5_w3, *FIB_IDEAL["w5_over_w3"]):
        conf += 0.15
        hits.append("wave5 ~0.618 x wave3")
    if w5_w1 is not None and _fib_hit(w5_w1, 0.618, 1.618):
        conf += 0.10
        hits.append("wave5 ~ wave1 magnitude")
    return min(conf, 1.0), hits


# =====================================================================
# PATTERN FITTERS — each returns a result dict or None
# =====================================================================

def _fit_impulse(zz: List[Tuple[int, float, str]], up: bool,
                 price: float) -> Optional[Dict]:
    """Full 5-wave impulse on the last 6 pivots (complete)."""
    if len(zz) < 6:
        return None
    tail = zz[-6:]
    want = ["L", "H", "L", "H", "L", "H"] if up else \
           ["H", "L", "H", "L", "H", "L"]
    if [t for _, _, t in tail] != want:
        return None
    p = [x[1] for x in tail]

    w1 = (p[1] - p[0]) if up else (p[0] - p[1])
    w3 = (p[3] - p[2]) if up else (p[2] - p[3])
    w5 = (p[5] - p[4]) if up else (p[4] - p[5])
    w2_ret = (p[1] - p[2]) / w1 if up and w1 > 0 else \
             ((p[2] - p[1]) / w1 if (not up) and w1 > 0 else None)
    w4_ret = (p[3] - p[4]) / w3 if up and w3 > 0 else \
             ((p[4] - p[3]) / w3 if (not up) and w3 > 0 else None)
    if w2_ret is None or w4_ret is None:
        return None

    # --- Elliott rules ---
    if not (0.10 <= w2_ret < 1.0):
        return None                                   # wave 2 rule
    if up and p[3] <= p[1]:
        return None                                   # wave 3 must exceed w1 top
    if (not up) and p[3] >= p[1]:
        return None
    tol = OVERLAP_TOLERANCE_PCT
    if up and p[4] <= p[1] * (1 - tol):
        return None                                   # wave 4 overlap rule
    if (not up) and p[4] >= p[1] * (1 + tol):
        return None
    if w3 < min(w1, w5) * WAVE3_SHORT_TOLERANCE:
        return None                                   # wave 3 not shortest
    if w5 <= 0:
        return None

    r31 = w3 / w1 if w1 > 0 else 0.0
    r53 = w5 / w3 if w3 > 0 else 0.0
    r51 = w5 / w1 if w1 > 0 else 0.0
    conf, hits = _impulse_confidence(w2_ret, r31, w4_ret, r53, r51)

    return {
        "pattern": "impulse_up" if up else "impulse_down",
        "current_wave": "5 (complete)",
        "wave_confidence": conf,
        "fib_hits": hits,
        "projection": None,
        "maturity": 1.0,
        "implication": ("Impulse complete - correction expected, "
                        "avoid fresh continuation entries"),
        "pivots": p,
    }


def _fit_partial_impulse(zz: List[Tuple[int, float, str]], up: bool,
                         price: float) -> Optional[Dict]:
    """Wave 4 confirmed on the last 5 pivots; wave 5 forming."""
    if len(zz) < 5:
        return None
    tail = zz[-5:]
    want = ["L", "H", "L", "H", "L"] if up else \
           ["H", "L", "H", "L", "H"]
    if [t for _, _, t in tail] != want:
        return None
    p = [x[1] for x in tail]

    w1 = (p[1] - p[0]) if up else (p[0] - p[1])
    w3 = (p[3] - p[2]) if up else (p[2] - p[3])
    w2_ret = (p[1] - p[2]) / w1 if up and w1 > 0 else \
             ((p[2] - p[1]) / w1 if (not up) and w1 > 0 else None)
    w4_ret = (p[3] - p[4]) / w3 if up and w3 > 0 else \
             ((p[4] - p[3]) / w3 if (not up) and w3 > 0 else None)
    if w2_ret is None or w4_ret is None:
        return None

    if not (0.10 <= w2_ret < 1.0):
        return None
    if up and p[3] <= p[1]:
        return None
    if (not up) and p[3] >= p[1]:
        return None
    tol = OVERLAP_TOLERANCE_PCT
    if up and p[4] <= p[1] * (1 - tol):
        return None
    if (not up) and p[4] >= p[1] * (1 + tol):
        return None

    r31 = w3 / w1 if w1 > 0 else 0.0
    conf, hits = _impulse_confidence(w2_ret, r31, w4_ret, base=0.4)
    if w3 < w1 * WAVE3_SHORT_TOLERANCE:
        # wave 3 suspiciously short -> weaker impulse reading
        conf = max(0.3, conf - 0.15)
        hits.append("wave3 short - reduced confidence")

    # Wave 5 projection: 0.618 x wave 3 from the wave-4 extreme
    projection = None
    maturity = None
    if w3 > 0:
        proj_len = WAVE5_PROJ_RATIO * w3
        projection = (p[4] + proj_len) if up else (p[4] - proj_len)
        remaining = (projection - price) if up else (price - projection)
        total = (projection - p[4]) if up else (p[4] - projection)
        if total and total > 0:
            maturity = max(0.0, min(1.5, 1.0 - remaining / total))

    in_wave5 = price > p[3] if up else price < p[3]
    if in_wave5:
        wave = "5"
        implication = ("Wave 5 in progress - final leg of the impulse")
    else:
        wave = "4"
        implication = ("Wave 4 pullback - wave 5 ahead, good R/R zone")

    return {
        "pattern": "partial_impulse_up" if up else "partial_impulse_down",
        "current_wave": wave,
        "wave_confidence": conf,
        "fib_hits": hits,
        "projection": float(projection) if projection else None,
        "maturity": maturity,
        "implication": implication,
        "pivots": p,
    }


def _fit_wave3(zz: List[Tuple[int, float, str]], up: bool,
               price: float) -> Optional[Dict]:
    """Wave 3 in progress: last 3 pivots [start, w1-top, w2-bottom] + breakout."""
    if len(zz) < 3:
        return None
    tail = zz[-3:]
    want = ["L", "H", "L"] if up else ["H", "L", "H"]
    if [t for _, _, t in tail] != want:
        return None
    p = [x[1] for x in tail]

    w1 = (p[1] - p[0]) if up else (p[0] - p[1])
    w2_ret = (p[1] - p[2]) / w1 if up and w1 > 0 else \
             ((p[2] - p[1]) / w1 if (not up) and w1 > 0 else None)
    if w2_ret is None:
        return None
    if not (0.10 <= w2_ret < 1.0):
        return None

    broke_out = price > p[1] if up else price < p[1]
    if not broke_out:
        return None  # price hasn't started wave 3 yet

    hits: List[str] = []
    conf = 0.4
    if _fib_hit(w2_ret, *FIB_IDEAL["w2"]):
        conf += 0.30
        hits.append("wave2 golden retrace (0.5-0.618)")
    elif _fib_hit(w2_ret, 0.382, 0.786):
        conf += 0.20
        hits.append("wave2 fib retrace (0.382-0.786)")
    # wave 3 projected target: w2 low + 1.618 x wave 1
    projection = (p[2] + 1.618 * w1) if up else (p[2] - 1.618 * w1)

    return {
        "pattern": "wave3_up" if up else "wave3_down",
        "current_wave": "3",
        "wave_confidence": min(conf, 1.0),
        "fib_hits": hits,
        "projection": float(projection),
        "maturity": None,
        "implication": "Wave 3 in progress - strongest phase of the cycle",
        "pivots": p,
    }


def _fit_abc(zz: List[Tuple[int, float, str]], after_up: bool,
             price: float) -> Optional[Dict]:
    """
    ABC correction. after_up=True -> ABC down after an up impulse
    (last 4 pivots H,L,H,L). after_up=False -> ABC up after a down
    impulse (last 4 pivots L,H,L,H).
    """
    if len(zz) < 4:
        return None
    tail = zz[-4:]
    want = ["H", "L", "H", "L"] if after_up else ["L", "H", "L", "H"]
    if [t for _, _, t in tail] != want:
        return None
    p = [x[1] for x in tail]

    if after_up:
        a_len = p[0] - p[1]          # A: down
        b_top = p[2]                 # B: up (must stay below origin)
        c_len = p[2] - p[3]          # C: down
        if a_len <= 0 or c_len <= 0:
            return None
        if p[2] > p[0] * 1.005:
            return None              # B exceeded origin -> new impulse up
        c_complete = p[3] <= p[1]    # C pierced below A low
    else:
        a_len = p[1] - p[0]          # A: up
        b_bottom = p[2]              # B: down (must stay above origin)
        c_len = p[3] - p[2]          # C: up
        if a_len <= 0 or c_len <= 0:
            return None
        if p[2] < p[0] * 0.995:
            return None              # B broke origin -> new impulse down
        c_complete = p[3] >= p[1]    # C exceeded A high

    c_over_a = c_len / a_len if a_len > 0 else 0.0
    hits: List[str] = []
    conf = 0.35
    if _fib_hit(c_over_a, *FIB_IDEAL["c_over_a"]):
        conf += 0.25
        hits.append("C ~ 1.0 x A")
    elif 1.4 <= c_over_a <= 1.8:
        conf += 0.15
        hits.append("C ~ 1.618 x A")
    elif 0.5 <= c_over_a < 0.8:
        conf += 0.05
        hits.append("C shallow vs A")

    if after_up:
        if c_complete and price > p[3]:
            wave, implication = "C (complete)", \
                "ABC correction complete - new cycle starting"
        else:
            wave, implication = "C", \
                "ABC correction still in progress - wait for completion"
    else:
        if c_complete and price < p[3]:
            wave, implication = "C (complete)", \
                "ABC bounce complete - downtrend resuming"
        else:
            wave, implication = "C", \
                "ABC bounce still in progress - wait for completion"

    return {
        "pattern": "correction_after_up" if after_up else "correction_after_down",
        "current_wave": wave,
        "wave_confidence": min(conf, 1.0),
        "fib_hits": hits,
        "projection": None,
        "maturity": None,
        "implication": implication,
        "pivots": p,
    }


# =====================================================================
# MAIN ENTRY
# =====================================================================

def detect_elliott_wave(df: pd.DataFrame, lookback: int = 150,
                        strength: int = 3) -> Dict:
    """
    Detect the current Elliott Wave position from the zigzag pivots.

    Tries the most specific pattern first: complete impulse (6 pivots),
    partial impulse (5), ABC correction (4), then wave-3 breakout (3).

    Returns an "unclear" result (pattern="unclear") when no structure
    fits - the confluence engine then applies NO adjustment.
    """
    unclear = {
        "pattern": "unclear",
        "current_wave": None,
        "wave_confidence": 0.0,
        "fib_hits": [],
        "projection": None,
        "maturity": None,
        "implication": "",
        "pivots": [],
    }
    if df is None or len(df) < 40:
        return unclear

    price = float(df["close"].iloc[-1])
    zz = _zigzag(df, lookback=lookback, strength=strength)
    if len(zz) < 3:
        return unclear

    fit = (_fit_impulse(zz, up=True, price=price)
           or _fit_impulse(zz, up=False, price=price)
           or _fit_partial_impulse(zz, up=True, price=price)
           or _fit_partial_impulse(zz, up=False, price=price)
           or _fit_abc(zz, after_up=True, price=price)
           or _fit_abc(zz, after_up=False, price=price)
           or _fit_wave3(zz, up=True, price=price)
           or _fit_wave3(zz, up=False, price=price))

    if fit is None:
        return unclear
    fit["wave_confidence"] = round(float(fit["wave_confidence"]), 3)
    if fit["projection"] is not None:
        fit["projection"] = float(fit["projection"])
    return fit
