"""
Tests for the Elliott Wave engine (v4).

Synthetic zigzag series with asymmetric wicks produce strict pivots, so
every classical pattern can be tested deterministically:

  impulse up (complete)   [104,100,112,105,124,116.5,129,126.5]
  partial impulse (w4)    [104,100,112,105,124,116.5,119]
  wave 3 breakout         [104,100,112,105,115]
  ABC after up impulse    [104,100,112,105,124,116.5,129,118,125,116,117.5]
  impulse down (complete) [125,129,116.5,124,105,112,100,102]
"""
import math
import pandas as pd
import pytest

from src.indicators.elliott import (
    detect_elliott_wave,
    _zigzag,
)


def trend_df(n=120, start=100.0, drift=0.004, amp=0.002):
    rows, price = [], start
    for i in range(n):
        change = drift + amp * math.sin(i * 1.7)
        close = price * (1 + change)
        high = max(price, close) * 1.001
        low = min(price, close) * 0.999
        rows.append({"open": price, "high": high, "low": low,
                     "close": close, "volume": 1000.0})
        price = close
    df = pd.DataFrame(rows)
    df.index = pd.date_range("2025-01-01", periods=len(df), freq="h")
    return df


def wave_df(points, bars=10):
    """Zigzag series; asymmetric wicks make turn bars strict pivots."""
    prices = []
    for k in range(len(points) - 1):
        a, b = points[k], points[k + 1]
        for j in range(bars):
            t = j / bars
            prices.append(a + (b - a) * t)
    prices.append(points[-1])
    rows, prev = [], prices[0]
    w = 0.0012
    for c in prices:
        o = prev
        if c >= o:
            hi, lo = c * (1 + w), o * (1 - w * 0.5)
        else:
            hi, lo = o * (1 + w * 0.5), c * (1 - w)
        rows.append({"open": o, "high": hi, "low": lo, "close": c,
                     "volume": 1000.0})
        prev = c
    df = pd.DataFrame(rows)
    df.index = pd.date_range("2025-01-01", periods=len(df), freq="h")
    return df


# ------------------------------------------------------------------
# Zigzag builder
# ------------------------------------------------------------------

def test_zigzag_alternates_and_orders():
    zz = _zigzag(wave_df([104, 100, 112, 105, 124, 116.5, 129, 126.5]))
    assert len(zz) >= 6
    types = [t for _, _, t in zz]
    # strictly alternating
    for a, b in zip(types, types[1:]):
        assert a != b


# ------------------------------------------------------------------
# Pattern detection
# ------------------------------------------------------------------

def test_complete_impulse_up_detected_with_fib_hits():
    r = detect_elliott_wave(wave_df([104, 100, 112, 105, 124, 116.5, 129, 126.5]))
    assert r["pattern"] == "impulse_up"
    assert r["current_wave"] == "5 (complete)"
    assert r["wave_confidence"] >= 0.4
    # classic fib relationships should be recognised on this clean zigzag
    assert any("wave2" in h for h in r["fib_hits"])
    assert any("wave3" in h for h in r["fib_hits"])
    assert r["maturity"] == 1.0


def test_complete_impulse_down_detected():
    r = detect_elliott_wave(wave_df([125, 129, 116.5, 124, 105, 112, 100.0, 102.0]))
    assert r["pattern"] == "impulse_down"
    assert r["current_wave"] == "5 (complete)"
    assert r["wave_confidence"] >= 0.4


def test_partial_impulse_wave4_with_projection():
    r = detect_elliott_wave(wave_df([104, 100, 112, 105, 124, 116.5, 119.0]))
    assert r["pattern"] == "partial_impulse_up"
    # price (119) is still below the w3 top (124) -> wave 4 zone
    assert r["current_wave"] == "4"
    # projection = w4 low + 0.618 x w3 length = 116.5 + 0.618*19
    assert r["projection"] == pytest.approx(116.5 + 0.618 * 19, rel=0.01)
    assert 0 <= r["maturity"] <= 1.5


def test_wave3_breakout_detected():
    r = detect_elliott_wave(wave_df([104, 100, 112, 105, 115.0], bars=12))
    assert r["pattern"] == "wave3_up"
    assert r["current_wave"] == "3"
    # projection = w2 low + 1.618 x w1 = 105 + 1.618*12
    assert r["projection"] == pytest.approx(105 + 1.618 * 12, rel=0.01)


def test_abc_correction_complete():
    r = detect_elliott_wave(
        wave_df([104, 100, 112, 105, 124, 116.5, 129, 118, 125, 116, 117.5]))
    assert r["pattern"] == "correction_after_up"
    assert r["current_wave"] == "C (complete)"
    assert r["wave_confidence"] >= 0.5


def test_flat_market_is_unclear():
    r = detect_elliott_wave(trend_df(drift=0.0, amp=0.004))
    assert r["pattern"] == "unclear"
    assert r["wave_confidence"] == 0.0


def test_short_data_is_unclear_not_crash():
    r = detect_elliott_wave(wave_df([100, 110, 105], bars=5))
    assert r["pattern"] == "unclear"


def test_result_is_json_serialisable():
    import json
    r = detect_elliott_wave(wave_df([104, 100, 112, 105, 124, 116.5, 129, 126.5]))
    json.dumps(r)  # must not raise
