"""
Fibonacci + Support/Resistance Entry/Exit Engine (v3)

Computes trade ENTRY and EXIT points from Fibonacci retracements /
extensions combined with structural support & resistance levels.

Core concepts:
  - Auto-detect the dominant impulse (swing low <-> swing high) and its
    direction from the most recent significant swing points in the window.
  - UP impulse (high formed AFTER low): pullback entries in the "golden
    pocket" (0.5 - 0.618 retracement), targets at Fibonacci extensions
    (1.272 / 1.414 / 1.618) and structural resistances.
  - DOWN impulse (low formed AFTER high): reversal entries near the bounce,
    targets at retracement levels of the down leg (0.382 / 0.5 / 0.618)
    and structural resistances. Mirror logic for bearish setups.
  - A Fibonacci level within tolerance of a support/resistance level forms
    a "confluence zone" - preferred for entries and strengthens targets.
  - Stop loss stays structure-based (swing + ATR buffer, clamped) so the
    validated v2 risk model is preserved: SL distance in [0.9, 2.2] x ATR,
    TP1 always satisfies the minimum R/R ratio.
"""
from typing import Dict, List, Optional, Tuple

# Fibonacci ratios
RETRACEMENT_RATIOS = (0.236, 0.382, 0.5, 0.618, 0.786)
EXTENSION_RATIOS = (1.272, 1.414, 1.618, 2.0)
GOLDEN_DEEP = 0.618   # deeper edge of the golden pocket
GOLDEN_SHALLOW = 0.5  # shallower edge of the golden pocket

# Confluence tolerance: fib level + S/R level within this % = one zone
CONFLUENCE_TOLERANCE_PCT = 0.6


# ============================================
# SWING DETECTION
# ============================================

def find_swing_points(df, lookback: int = 100,
                      strength: int = 2) -> Tuple[List[Tuple[int, float]],
                                                  List[Tuple[int, float]]]:
    """
    Detect fractal swing highs/lows (price higher/lower than `strength`
    bars on each side). Returns (swing_highs, swing_lows) as lists of
    (index_in_window, price), oldest first.
    """
    window = df.iloc[-lookback:]
    highs = window["high"].values
    lows = window["low"].values
    n = len(highs)
    swing_highs: List[Tuple[int, float]] = []
    swing_lows: List[Tuple[int, float]] = []
    for i in range(strength, n - strength):
        if all(highs[i] > highs[i - j] for j in range(1, strength + 1)) and \
           all(highs[i] > highs[i + j] for j in range(1, strength + 1)):
            swing_highs.append((i, float(highs[i])))
        if all(lows[i] < lows[i - j] for j in range(1, strength + 1)) and \
           all(lows[i] < lows[i + j] for j in range(1, strength + 1)):
            swing_lows.append((i, float(lows[i])))
    return swing_highs, swing_lows


# ============================================
# FIBONACCI LEVELS
# ============================================

def compute_fibonacci_levels(df, lookback: int = 100) -> Dict:
    """
    Compute the active Fibonacci map from the dominant impulse.

    Returns {} when the market has no measurable range.

    Keys:
      impulse      "up" if swing high formed after swing low, else "down"
      swing_high / swing_low / range
      retracements {ratio: price} - pullback levels of the impulse
      extensions   {ratio: price} - continuation targets beyond the impulse
      golden_zone  {"low": price, "high": price} - 0.5..0.618 pocket
    """
    if df is None or len(df) < 20:
        return {}

    window = df.iloc[-lookback:]
    high_arr = window["high"].values
    low_arr = window["low"].values

    swing_high_idx = int(high_arr.argmax())
    swing_low_idx = int(low_arr.argmin())
    swing_high = float(high_arr[swing_high_idx])
    swing_low = float(low_arr[swing_low_idx])
    rng = swing_high - swing_low
    if rng <= 0:
        return {}

    impulse = "up" if swing_high_idx > swing_low_idx else "down"

    if impulse == "up":
        # Retracement: price pulls back DOWN from the high
        retracements = {str(r): swing_high - r * rng for r in RETRACEMENT_RATIOS}
        # Extensions: continuation targets ABOVE the high
        extensions = {str(e): swing_low + e * rng for e in EXTENSION_RATIOS}
        golden_zone = {
            "low": swing_high - GOLDEN_DEEP * rng,     # 0.618 level (lower)
            "high": swing_high - GOLDEN_SHALLOW * rng, # 0.5 level (upper)
        }
    else:
        # Retracement: price pulls back UP from the low
        retracements = {str(r): swing_low + r * rng for r in RETRACEMENT_RATIOS}
        # Extensions: continuation targets BELOW the low (bearish targets)
        extensions = {str(e): swing_high - e * rng for e in EXTENSION_RATIOS}
        golden_zone = {
            "low": swing_low + GOLDEN_SHALLOW * rng,   # 0.5 level (lower)
            "high": swing_low + GOLDEN_DEEP * rng,     # 0.618 level (upper)
        }

    return {
        "impulse": impulse,
        "swing_high": swing_high,
        "swing_low": swing_low,
        "range": float(rng),
        "retracements": {k: float(v) for k, v in retracements.items()},
        "extensions": {k: float(v) for k, v in extensions.items()},
        "golden_zone": {"low": float(golden_zone["low"]),
                         "high": float(golden_zone["high"])},
    }


# ============================================
# CONFLUENCE (Fibonacci x S/R)
# ============================================

def fib_sr_confluence(levels: Dict[str, float], sr_levels: List[Optional[float]],
                      tolerance_pct: float = CONFLUENCE_TOLERANCE_PCT) -> Dict[str, float]:
    """
    Map each Fibonacci level to a support/resistance level when the two are
    within tolerance_pct of each other. Returns {fib_key: sr_price}.
    """
    clean_sr = [float(s) for s in (sr_levels or []) if s]
    out: Dict[str, float] = {}
    for key, price in (levels or {}).items():
        if not price:
            continue
        best = None
        for sr in clean_sr:
            if abs(sr - price) / price * 100 <= tolerance_pct:
                if best is None or abs(sr - price) < abs(best - price):
                    best = sr
        if best is not None:
            out[key] = best
    return out


def _has_sr_confluence(price: float, sr_levels: List[Optional[float]],
                       tolerance_pct: float = CONFLUENCE_TOLERANCE_PCT) -> bool:
    for sr in (sr_levels or []):
        if sr and abs(float(sr) - price) / price * 100 <= tolerance_pct:
            return True
    return False


# ============================================
# ENTRY / EXIT COMPUTATION
# ============================================

def compute_entry_exit(direction: str, current_price: float, atr_val: float,
                       sr: Dict, fib: Dict, swing_low: float, swing_high: float,
                       min_rr: float = 1.5) -> Dict:
    """
    Fibonacci + S/R based entry and exit points.

    Parameters:
      direction     "bullish" | "bearish" | "neutral"
      current_price last close
      atr_val       ATR(14) value in price units
      sr            result of find_support_resistance() (supports nearest-first)
      fib           result of compute_fibonacci_levels() (may be {})
      swing_low / swing_high   10-bar structural levels (fallback + SL anchor)
      min_rr        minimum R/R enforced on TP1

    Returns dict with:
      entry_price, entry_type ("market"|"limit"), entry_zone {low, high},
      stop_loss, take_profit (TP1), take_profit_2 (TP2),
      risk_reward_ratio (TP1-based), tp2_rr,
      sl_distance_pct, tp_distance_pct, tp2_distance_pct,
      entry_label, tp1_label, tp2_label, fib_notes
    """
    notes: List[str] = []
    supports = list(sr.get("supports", []) or [])
    resistances = list(sr.get("resistances", []) or [])

    # ---------------- Fallback (neutral / no fib map) ----------------
    if not fib or direction == "neutral":
        return _fallback_entry_exit(direction, current_price, atr_val,
                                     supports, resistances,
                                     swing_low, swing_high, min_rr)

    impulse = fib["impulse"]
    rng = fib["range"]
    # Golden pocket edges (used by both bullish and bearish branches)
    gz_low = fib["golden_zone"]["low"]
    gz_high = fib["golden_zone"]["high"]

    # ---------------- Bullish ----------------
    if direction == "bullish":
        # --- Stop loss: below structure, clamped (v2 risk model) ---
        struct_dist = (current_price - swing_low) + 0.4 * atr_val
        sl_dist = min(max(struct_dist, 0.9 * atr_val), 2.2 * atr_val)
        stop_loss = current_price - sl_dist
        notes.append(f"SL below 10-bar swing low {swing_low:.6g} + 0.4 ATR buffer")

        # --- Entry zone: golden pocket ---
        if impulse == "up":
            entry_zone = {"low": gz_low, "high": gz_high}
            # pocket fib levels (0.5 and 0.618) - possible confluence anchors
            pocket_levels = {k: v for k, v in fib["retracements"].items()
                             if gz_low <= v <= gz_high}
            if current_price <= gz_high:
                # price already inside (or below) the golden pocket -> market
                entry_type, entry_price = "market", current_price
                entry_label = "Market - price in Fibonacci golden pocket"
            else:
                entry_type = "limit"
                conf = fib_sr_confluence(pocket_levels, supports)
                if conf:
                    # deepest confluent level in the pocket (best R/R)
                    entry_price = max(conf.values())
                    entry_label = ("Limit - Fib " +
                                   "/".join(conf.keys()) +
                                   " x support confluence")
                    notes.append("Entry at Fib x support confluence zone")
                else:
                    entry_price = gz_low  # 0.618 level = best R/R in pocket
                    entry_label = "Limit - Fibonacci 0.618 golden pocket"
        else:
            # DOWN impulse + bullish reversal: buy the bounce (market),
            # pullback zone = shallow retracements of the down leg
            zone_low = fib["retracements"]["0.236"]
            zone_high = fib["retracements"]["0.382"]
            entry_zone = {"low": zone_low, "high": zone_high}
            entry_type, entry_price = "market", current_price
            entry_label = "Market - reversal from swing low (discount zone)"
            if _has_sr_confluence(current_price, supports):
                entry_label += " + support confluence"
                notes.append("Reversal entry coincides with support level")

        # --- Targets: first structure levels above price ---
        targets: List[Tuple[float, str]] = []
        if impulse == "up":
            if fib["swing_high"] > current_price * 1.001:
                targets.append((fib["swing_high"], "Fib 1.0 (swing high)"))
            for e in ("1.272", "1.414", "1.618"):
                p = fib["extensions"][e]
                if p > current_price * 1.001:
                    targets.append((p, f"Fib {e} extension"))
        else:
            for r in ("0.382", "0.5", "0.618"):
                p = fib["retracements"][r]
                if p > current_price * 1.001:
                    targets.append((p, f"Fib {r} retracement"))
        for i, r_lvl in enumerate(resistances):
            if r_lvl > current_price * 1.001:
                targets.append((float(r_lvl), f"Resistance {i + 1}"))
        targets.sort(key=lambda t: t[0])

        tp1, tp1_label, tp2, tp2_label = _pick_targets(
            current_price, sl_dist, targets, resistances, min_rr, below=False)

        tp1_dist = tp1 - current_price
        tp2_dist = tp2 - current_price
        result = {
            "entry_price": float(entry_price),
            "entry_type": entry_type,
            "entry_zone": {"low": float(entry_zone["low"]),
                            "high": float(entry_zone["high"])},
            "stop_loss": float(stop_loss),
            "take_profit": float(tp1),
            "take_profit_2": float(tp2),
            "risk_reward_ratio": float(tp1_dist / sl_dist) if sl_dist > 0 else 0.0,
            "tp2_rr": float(tp2_dist / sl_dist) if sl_dist > 0 else 0.0,
            "sl_distance_pct": float(sl_dist / current_price * 100),
            "tp_distance_pct": float(tp1_dist / current_price * 100),
            "tp2_distance_pct": float(tp2_dist / current_price * 100),
            "entry_label": entry_label,
            "tp1_label": tp1_label,
            "tp2_label": tp2_label,
            "fib_notes": notes,
        }
        return result

    # ---------------- Bearish ----------------
    # --- Stop loss: above structure, clamped (v2 risk model) ---
    struct_dist = (swing_high - current_price) + 0.4 * atr_val
    sl_dist = min(max(struct_dist, 0.9 * atr_val), 2.2 * atr_val)
    stop_loss = current_price + sl_dist
    notes.append(f"SL above 10-bar swing high {swing_high:.6g} + 0.4 ATR buffer")

    # --- Entry zone ---
    if impulse == "down":
        entry_zone = {"low": gz_low, "high": gz_high}
        pocket_levels = {k: v for k, v in fib["retracements"].items()
                         if gz_low <= v <= gz_high}
        if current_price >= gz_high:
            # price pulled back into/above the golden pocket -> short now
            entry_type, entry_price = "market", current_price
            entry_label = "Market - price in Fibonacci golden pocket"
        else:
            entry_type = "limit"
            conf = fib_sr_confluence(pocket_levels, resistances)
            if conf:
                entry_price = max(conf.values())
                entry_label = ("Limit - Fib " +
                               "/".join(conf.keys()) +
                               " x resistance confluence")
                notes.append("Short entry at Fib x resistance confluence zone")
            else:
                entry_price = gz_high  # 0.618 level = best R/R short entry
                entry_label = "Limit - Fibonacci 0.618 golden pocket"
    else:
        # UP impulse + bearish reversal: short near the top (market)
        zone_low = fib["retracements"]["0.618"]
        zone_high = fib["retracements"]["0.786"]
        entry_zone = {"low": zone_low, "high": zone_high}
        entry_type, entry_price = "market", current_price
        entry_label = "Market - reversal from swing high (premium zone)"
        if _has_sr_confluence(current_price, resistances):
            entry_label += " + resistance confluence"
            notes.append("Reversal entry coincides with resistance level")

    # --- Targets below price ---
    targets = []
    if impulse == "down":
        if fib["swing_low"] < current_price * 0.999:
            targets.append((fib["swing_low"], "Fib 1.0 (swing low)"))
        for e in ("1.272", "1.414", "1.618"):
            p = fib["extensions"][e]
            if p < current_price * 0.999:
                targets.append((p, f"Fib {e} extension"))
    else:
        for r in ("0.382", "0.5", "0.618"):
            p = fib["retracements"][r]
            if p < current_price * 0.999:
                targets.append((p, f"Fib {r} retracement"))
    for i, s_lvl in enumerate(supports):
        if s_lvl < current_price * 0.999:
            targets.append((float(s_lvl), f"Support {i + 1}"))
    targets.sort(key=lambda t: t[0], reverse=True)  # closest below first

    tp1, tp1_label, tp2, tp2_label = _pick_targets(
        current_price, sl_dist, targets, supports, min_rr, below=True)

    tp1_dist = current_price - tp1
    tp2_dist = current_price - tp2
    return {
        "entry_price": float(entry_price),
        "entry_type": entry_type,
        "entry_zone": {"low": float(entry_zone["low"]),
                        "high": float(entry_zone["high"])},
        "stop_loss": float(stop_loss),
        "take_profit": float(tp1),
        "take_profit_2": float(tp2),
        "risk_reward_ratio": float(tp1_dist / sl_dist) if sl_dist > 0 else 0.0,
        "tp2_rr": float(tp2_dist / sl_dist) if sl_dist > 0 else 0.0,
        "sl_distance_pct": float(sl_dist / current_price * 100),
        "tp_distance_pct": float(tp1_dist / current_price * 100),
        "tp2_distance_pct": float(tp2_dist / current_price * 100),
        "entry_label": entry_label,
        "tp1_label": tp1_label,
        "tp2_label": tp2_label,
        "fib_notes": notes,
    }


def _pick_targets(current_price: float, sl_dist: float,
                  targets: List[Tuple[float, str]],
                  sr_levels: List[Optional[float]], min_rr: float,
                  below: bool) -> Tuple[float, str, float, str]:
    """
    Pick TP1 / TP2 from structure targets.
    TP1 = first target that satisfies min R/R (falls back to a pure R/R
    multiple when no structure target is far enough) - preserves the v2
    guarantee that TP1 always satisfies MIN_RR_RATIO.
    TP2 = next target beyond TP1 (capped at 8x SL distance).
    Confluence with S/R levels is appended to the labels.
    """
    cap_mult_1, cap_mult_2 = 5.0, 8.0

    def _confl(price: float, label: str) -> str:
        if _has_sr_confluence(price, sr_levels):
            return label + " + S/R confluence"
        return label

    tp1 = tp1_label = None
    for t, label in targets:
        dist = (current_price - t) if below else (t - current_price)
        if dist >= min_rr * sl_dist:
            tp1, tp1_label = t, _confl(t, label)
            break
    if tp1 is None:
        dist = min_rr * sl_dist
        tp1 = (current_price - dist) if below else (current_price + dist)
        tp1_label = "Min R/R target"

    # TP2: next structure target beyond TP1
    tp2 = tp2_label = None
    for t, label in targets:
        if below:
            if t < tp1 - 1e-12:
                tp2, tp2_label = t, _confl(t, label)
                break
        else:
            if t > tp1 + 1e-12:
                tp2, tp2_label = t, _confl(t, label)
                break
    if tp2 is None:
        ext = tp1 if tp1_label != "Min R/R target" else current_price
        base_dist = abs(current_price - ext) or sl_dist * min_rr
        tp2_dist = min(base_dist * 1.618, cap_mult_2 * sl_dist)
        tp2 = (current_price - tp2_dist) if below else (current_price + tp2_dist)
        tp2_label = "Fib extension (projected)"

    # Sanity caps
    if below:
        max_d1 = cap_mult_1 * sl_dist
        if current_price - tp1 > max_d1:
            tp1 = current_price - max_d1
        if current_price - tp2 > cap_mult_2 * sl_dist:
            tp2 = current_price - cap_mult_2 * sl_dist
        if tp2 >= tp1:
            tp2 = tp1 - sl_dist
        tp2 = max(tp2, 1e-12)
    else:
        max_d1 = cap_mult_1 * sl_dist
        if tp1 - current_price > max_d1:
            tp1 = current_price + max_d1
        if tp2 - current_price > cap_mult_2 * sl_dist:
            tp2 = current_price + cap_mult_2 * sl_dist
        if tp2 <= tp1:
            tp2 = tp1 + sl_dist
    return float(tp1), tp1_label, float(tp2), tp2_label


def _fallback_entry_exit(direction: str, current_price: float, atr_val: float,
                          supports: List, resistances: List,
                          swing_low: float, swing_high: float,
                          min_rr: float) -> Dict:
    """
    v2 structure-based fallback (no Fibonacci map / neutral market).
    SL at swing +/- ATR buffer clamped [0.9, 2.2]xATR,
    TP at nearest S/R with enforced minimum R/R.
    """
    if direction == "bearish":
        struct_dist = (swing_high - current_price) + 0.4 * atr_val
        sl_dist = min(max(struct_dist, 0.9 * atr_val), 2.2 * atr_val)
        stop_loss = current_price + sl_dist
        s_lvl = supports[0] if supports else None
        tp_struct = (s_lvl - current_price) if s_lvl and s_lvl < current_price else None
        tp_dist = max(tp_struct, min_rr * sl_dist) if tp_struct else min_rr * sl_dist
        tp_dist = min(tp_dist, 5.0 * sl_dist)
        take_profit = current_price - tp_dist
        take_profit_2 = current_price - min(tp_dist * 1.618, 8.0 * sl_dist)
        take_profit_2 = max(take_profit_2, 1e-12)
        entry_type, entry_price = "market", current_price
        entry_zone = {"low": current_price - 0.5 * sl_dist,
                       "high": current_price + 0.5 * sl_dist}
        entry_label = "Market - structural reversal entry"
        tp1_label = "Nearest support" if tp_struct else "Min R/R target"
        tp2_label = "Projected extension"
    elif direction == "bullish":
        struct_dist = (current_price - swing_low) + 0.4 * atr_val
        sl_dist = min(max(struct_dist, 0.9 * atr_val), 2.2 * atr_val)
        stop_loss = current_price - sl_dist
        r_lvl = resistances[0] if resistances else None
        tp_struct = (r_lvl - current_price) if r_lvl and r_lvl > current_price else None
        tp_dist = max(tp_struct, min_rr * sl_dist) if tp_struct else min_rr * sl_dist
        tp_dist = min(tp_dist, 5.0 * sl_dist)
        take_profit = current_price + tp_dist
        take_profit_2 = current_price + min(tp_dist * 1.618, 8.0 * sl_dist)
        entry_type, entry_price = "market", current_price
        entry_zone = {"low": current_price - 0.5 * sl_dist,
                       "high": current_price + 0.5 * sl_dist}
        entry_label = "Market - structural breakout entry"
        tp1_label = "Nearest resistance" if tp_struct else "Min R/R target"
        tp2_label = "Projected extension"
    else:
        sl_dist = 1.5 * atr_val
        stop_loss = current_price - sl_dist
        tp_dist = 1.5 * sl_dist
        take_profit = current_price + tp_dist
        take_profit_2 = current_price + tp_dist * 1.5
        entry_type, entry_price = "market", current_price
        entry_zone = {"low": current_price - 0.5 * sl_dist,
                       "high": current_price + 0.5 * sl_dist}
        entry_label = "Market - neutral zone"
        tp1_label = "Neutral target"
        tp2_label = "Extended target"

    return {
        "entry_price": float(entry_price),
        "entry_type": entry_type,
        "entry_zone": {"low": float(entry_zone["low"]),
                        "high": float(entry_zone["high"])},
        "stop_loss": float(stop_loss),
        "take_profit": float(take_profit),
        "take_profit_2": float(take_profit_2),
        "risk_reward_ratio": float(tp_dist / sl_dist) if sl_dist > 0 else 0.0,
        "tp2_rr": float(abs(take_profit_2 - current_price) / sl_dist) if sl_dist > 0 else 0.0,
        "sl_distance_pct": float(sl_dist / current_price * 100) if current_price > 0 else 0.0,
        "tp_distance_pct": float(tp_dist / current_price * 100) if current_price > 0 else 0.0,
        "tp2_distance_pct": float(abs(take_profit_2 - current_price) / current_price * 100) if current_price > 0 else 0.0,
        "entry_label": entry_label,
        "tp1_label": tp1_label,
        "tp2_label": tp2_label,
        "fib_notes": [],
    }
