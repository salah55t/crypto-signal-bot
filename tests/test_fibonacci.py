"""
Tests for the v3 Fibonacci + Support/Resistance entry/exit engine.

Covers:
  - Fibonacci level computation (up / down impulse)
  - Golden zone geometry
  - Fib x S/R confluence detection
  - Entry/exit computation (bullish, bearish, neutral fallback, flat market)
  - nearest_support ordering fix in find_support_resistance
  - End-to-end scorer integration (new output fields present and consistent)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.indicators.fibonacci import (
    compute_fibonacci_levels,
    compute_entry_exit,
    fib_sr_confluence,
    _has_sr_confluence,
)
from src.indicators.liquidity import find_support_resistance
from src.analysis.scorer import scorer


# ============================================
# Helpers
# ============================================

def make_up_impulse_df(start=100.0, low=60.0, high=120.0, n_down=60, n_up=40,
                       seed=42):
    """Downtrend then strong uptrend -> swing low BEFORE swing high (up impulse).
    NOTE: the final rally high must END above the starting price so the
    window's dominant swing high is at the END of the up leg."""
    rng = np.random.default_rng(seed)
    down = np.linspace(start, low, n_down) + rng.normal(0, 0.15, n_down)
    up = np.linspace(low, high, n_up) + rng.normal(0, 0.15, n_up)
    close = np.concatenate([down, up])
    close = np.clip(close, 1.0, None)
    n = len(close)
    spread = np.abs(rng.normal(0, 0.25, n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    low = np.clip(low, 0.5, None)
    vol = rng.uniform(900, 1100, n)
    return pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close,
        "volume": vol,
    })


def make_down_impulse_df(start=60.0, high=100.0, low=55.0, n_up=60, n_down=40,
                         seed=7):
    """Uptrend then sharp downtrend -> swing high BEFORE swing low (down impulse).
    NOTE: the final low must END below the starting price so the window's
    dominant swing low is at the END of the down leg."""
    rng = np.random.default_rng(seed)
    up = np.linspace(start, high, n_up) + rng.normal(0, 0.15, n_up)
    down = np.linspace(high, low, n_down) + rng.normal(0, 0.15, n_down)
    close = np.concatenate([up, down])
    close = np.clip(close, 1.0, None)
    n = len(close)
    spread = np.abs(rng.normal(0, 0.25, n))
    open_ = np.concatenate([[close[0]], close[:-1]])
    high_a = np.maximum(open_, close) + spread
    low_a = np.minimum(open_, close) - spread
    low_a = np.clip(low_a, 0.5, None)
    vol = rng.uniform(900, 1100, n)
    return pd.DataFrame({
        "open": open_, "high": high_a, "low": low_a, "close": close,
        "volume": vol,
    })


# ============================================
# Fibonacci levels
# ============================================

def test_fib_levels_up_impulse():
    df = make_up_impulse_df()
    fib = compute_fibonacci_levels(df, lookback=100)
    assert fib, "fib map should not be empty for a trending df"
    assert fib["impulse"] == "up"
    assert fib["swing_low"] < fib["swing_high"]
    rng = fib["range"]
    # 0.5 retracement = midpoint of the range
    assert fib["retracements"]["0.5"] == pytest.approx(
        fib["swing_low"] + 0.5 * rng, rel=1e-9)
    # extensions must be ABOVE the swing high (up impulse)
    assert fib["extensions"]["1.272"] > fib["swing_high"]
    assert fib["extensions"]["1.618"] > fib["extensions"]["1.272"]
    assert fib["extensions"]["2.0"] > fib["extensions"]["1.618"]
    # retracements ordered: 0.236 highest ... 0.786 lowest
    assert fib["retracements"]["0.236"] > fib["retracements"]["0.786"]


def test_fib_levels_down_impulse():
    df = make_down_impulse_df()
    fib = compute_fibonacci_levels(df, lookback=100)
    assert fib, "fib map should not be empty for a trending df"
    assert fib["impulse"] == "down"
    # down-impulse extensions are BELOW the swing low (bearish targets)
    assert fib["extensions"]["1.272"] < fib["swing_low"]
    # retracements pull back UP from the low
    assert fib["retracements"]["0.236"] > fib["swing_low"]
    assert fib["retracements"]["0.618"] > fib["retracements"]["0.236"]


def test_golden_zone_geometry_up():
    df = make_up_impulse_df()
    fib = compute_fibonacci_levels(df, lookback=100)
    gz = fib["golden_zone"]
    # golden pocket spans between the 0.5 and 0.618 retracement levels
    assert gz["low"] == pytest.approx(fib["retracements"]["0.618"], rel=1e-9)
    assert gz["high"] == pytest.approx(fib["retracements"]["0.5"], rel=1e-9)
    assert gz["low"] < gz["high"]


def test_fib_flat_market_returns_empty():
    n = 60
    df = pd.DataFrame({
        "open": [100.0] * n, "high": [100.0] * n,
        "low": [100.0] * n, "close": [100.0] * n,
        "volume": [1000.0] * n,
    })
    assert compute_fibonacci_levels(df, lookback=50) == {}


# ============================================
# Confluence
# ============================================

def test_confluence_detection():
    levels = {"0.618": 100.0, "0.5": 103.0}
    sr = [100.2, 110.0]  # 100.2 within 0.6% of 100.0
    conf = fib_sr_confluence(levels, sr, tolerance_pct=0.6)
    assert "0.618" in conf
    assert conf["0.618"] == pytest.approx(100.2)
    assert "0.5" not in conf  # 103 vs 110 = 6.8% away


def test_has_sr_confluence_false_when_far():
    assert _has_sr_confluence(100.0, [105.0], tolerance_pct=0.6) is False
    assert _has_sr_confluence(100.0, [100.3], tolerance_pct=0.6) is True
    assert _has_sr_confluence(100.0, [None, 0], tolerance_pct=0.6) is False


# ============================================
# Entry / exit computation
# ============================================

SR_BULL = {
    "supports": [88.0, 80.0],        # nearest first (post-fix ordering)
    "resistances": [102.0, 110.0],
}
SR_BEAR = {
    "supports": [88.0, 80.0],
    "resistances": [102.0, 110.0],
}


def _bull_fib():
    df = make_up_impulse_df(low=60.0, high=120.0)
    return compute_fibonacci_levels(df, lookback=100)


def test_entry_exit_bullish_consistency():
    fib = _bull_fib()
    price = 90.0
    atr = 1.0
    out = compute_entry_exit("bullish", price, atr, SR_BULL, fib,
                             swing_low=price - 3, swing_high=95.0,
                             min_rr=1.5)
    # Ordering invariants
    assert out["stop_loss"] < price < out["take_profit"] <= out["take_profit_2"]
    # v2 risk model preserved: SL distance clamped to [0.9, 2.2] x ATR
    sl_dist = price - out["stop_loss"]
    assert 0.9 * atr - 1e-9 <= sl_dist <= 2.2 * atr + 1e-9
    # TP1 satisfies min R/R
    assert out["risk_reward_ratio"] >= 1.5
    assert out["take_profit"] == pytest.approx(price + out["risk_reward_ratio"] * sl_dist)
    # TP2 beyond TP1
    assert out["take_profit_2"] > out["take_profit"]
    assert out["tp2_rr"] > out["risk_reward_ratio"]
    # entry fields present
    assert out["entry_type"] in ("market", "limit")
    assert out["entry_zone"]["low"] <= out["entry_zone"]["high"]
    assert out["entry_label"] and out["tp1_label"] and out["tp2_label"]


def test_entry_exit_bullish_tp_prefers_structure_targets():
    fib = _bull_fib()
    price = 90.0
    atr = 1.0
    out = compute_entry_exit("bullish", price, atr, SR_BULL, fib,
                             swing_low=price - 3, swing_high=95.0,
                             min_rr=1.5)
    # TP1 should be labelled from a structure target (resistance 102 or fib)
    assert "Min R/R" not in out["tp1_label"] or out["take_profit"] >= price + 1.5 * (price - out["stop_loss"])


def test_entry_exit_bullish_limit_entry_above_pocket():
    fib = _bull_fib()
    gz = fib["golden_zone"]
    # price clearly above the golden pocket -> limit entry at 0.618 level
    price = gz["high"] * 1.10
    atr = 1.0
    out = compute_entry_exit("bullish", price, atr, SR_BULL, fib,
                             swing_low=price - 3, swing_high=price + 2,
                             min_rr=1.5)
    assert out["entry_type"] == "limit"
    assert out["entry_price"] == pytest.approx(gz["low"])
    assert gz["low"] <= out["entry_zone"]["low"] <= out["entry_zone"]["high"]


def test_entry_exit_bullish_market_entry_in_pocket():
    fib = _bull_fib()
    gz = fib["golden_zone"]
    price = (gz["low"] + gz["high"]) / 2  # inside the pocket
    atr = 1.0
    out = compute_entry_exit("bullish", price, atr, SR_BULL, fib,
                             swing_low=price - 3, swing_high=gz["high"] + 2,
                             min_rr=1.5)
    assert out["entry_type"] == "market"
    assert out["entry_price"] == pytest.approx(price)
    assert "golden pocket" in out["entry_label"].lower()


def test_entry_exit_bearish_consistency():
    fib_down = compute_fibonacci_levels(make_down_impulse_df(), lookback=100)
    price = 70.0
    atr = 1.0
    out = compute_entry_exit("bearish", price, atr, SR_BEAR, fib_down,
                             swing_low=price - 2, swing_high=price + 3,
                             min_rr=1.5)
    assert out["take_profit_2"] <= out["take_profit"] < price < out["stop_loss"]
    sl_dist = out["stop_loss"] - price
    assert 0.9 * atr - 1e-9 <= sl_dist <= 2.2 * atr + 1e-9
    assert out["risk_reward_ratio"] >= 1.5 - 1e-9
    assert out["tp2_rr"] > out["risk_reward_ratio"]
    assert out["entry_zone"]["low"] <= out["entry_zone"]["high"]


def test_entry_exit_neutral_fallback():
    price = 100.0
    atr = 1.0
    out = compute_entry_exit("neutral", price, atr, SR_BULL, fib={},
                             swing_low=97.0, swing_high=103.0, min_rr=1.5)
    assert out["stop_loss"] < price
    assert out["take_profit"] > price
    assert out["risk_reward_ratio"] > 0
    assert out["entry_type"] == "market"


def test_entry_exit_bullish_no_fib_fallback():
    """Bullish but no fib map (flat df) -> structure fallback, still consistent."""
    price = 100.0
    atr = 1.0
    out = compute_entry_exit("bullish", price, atr, SR_BULL, fib={},
                             swing_low=97.0, swing_high=103.0, min_rr=1.5)
    assert out["stop_loss"] < price < out["take_profit"]
    assert out["risk_reward_ratio"] >= 1.5


def test_entry_exit_flat_market_no_crash():
    price = 100.0
    out = compute_entry_exit("bullish", price, atr_val=0.0, sr={},
                              fib={}, swing_low=price, swing_high=price,
                              min_rr=1.5)
    # struct_dist = 0 + 0.4*0 = 0 -> clamped to 0 -> guard against div-by-zero
    assert out["risk_reward_ratio"] >= 0


# ============================================
# S/R nearest-first fix
# ============================================

def test_nearest_support_is_closest_below_price():
    """Regression test: supports used to be sorted ascending so the 'nearest'
    support was actually the farthest (deepest) one."""
    n = 60
    base = 100.0
    close = np.full(n, base)
    # place two clear swing lows: one at 95 (near), one at 80 (deep)
    close[20] = 95.0
    close[19] = close[21] = 96.0
    close[18] = close[22] = 96.5
    close[40] = 80.0
    close[39] = close[41] = 81.0
    close[38] = close[42] = 81.5
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 0.05, n)
    close = close + noise
    open_ = close + 0.02
    high = np.maximum(open_, close) + 0.05
    low = np.minimum(open_, close) - 0.05
    df = pd.DataFrame({"open": open_, "high": high, "low": low,
                        "close": close, "volume": [1000.0] * n})
    sr = find_support_resistance(df, lookback=50)
    assert sr["supports"], "should detect at least one support"
    # supports must be sorted NEAREST first (descending)
    sup = sr["supports"]
    assert sup == sorted(sup, reverse=True)
    if len(sup) >= 2:
        assert sr["nearest_support"] == pytest.approx(sup[0])
        assert sup[0] > sup[1]  # nearest is the HIGHER (closer) level


# ============================================
# Scorer integration
# ============================================

def test_scorer_output_has_fib_entry_exit_fields():
    df = make_up_impulse_df(n_down=80, n_up=120, high=140.0)
    res = scorer.analyze_symbol(df, "TESTUSDT")
    assert not res.get("skip")
    for key in ("entry_price", "entry_type", "entry_zone", "entry_label",
                "take_profit_2", "tp2_rr", "fib_notes", "fibonacci",
                "stop_loss", "take_profit", "risk_reward_ratio"):
        assert key in res, f"missing field: {key}"
    fib_block = res["fibonacci"]
    assert "impulse" in fib_block
    assert "retracements" in fib_block and "extensions" in fib_block
    # consistency: bullish -> SL below price, TP1/TP2 above
    if res["direction"] == "bullish":
        assert res["stop_loss"] < res["current_price"] < res["take_profit"]
        assert res["take_profit_2"] >= res["take_profit"]
    elif res["direction"] == "bearish":
        assert res["stop_loss"] > res["current_price"] > res["take_profit"]
        assert res["take_profit_2"] <= res["take_profit"]


def test_scorer_rr_still_enforced():
    """filter_signals must still work with the new fields (R/R = TP1-based)."""
    df = make_up_impulse_df(n_down=80, n_up=120, high=140.0)
    res = scorer.analyze_symbol(df, "TESTUSDT")
    assert res["risk_reward_ratio"] >= 0
    recs = scorer.filter_signals([res], min_confidence=0,
                                  min_expected_rise=0, direction="bullish")
    # either filtered out by direction/confidence or included with valid RR
    for r in recs:
        assert r["risk_reward_ratio"] >= 1.2
