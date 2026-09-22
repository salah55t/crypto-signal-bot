"""
Tests for the Ichimoku regime layer (v4).

Synthetic trend data with alternating small oscillation produces clean
above-cloud / below-cloud / inside-cloud regimes.
"""
import math
import pandas as pd
import pytest

from src.indicators.ichimoku import (
    ichimoku_lines,
    ichimoku_state,
    MIN_BARS,
)


def trend_df(n=120, start=100.0, drift=0.004, amp=0.002):
    """Trending series with a mild sine wiggle (deterministic)."""
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


def test_lines_columns_present():
    df = trend_df()
    lines = ichimoku_lines(df)
    for col in ("tenkan", "kijun", "span_a", "span_b", "chikou"):
        assert col in lines.columns
    # Early rows are NaN (rolling windows + displacement), tail is defined
    assert lines["span_b"].iloc[-1] == lines["span_b"].iloc[-1]  # not NaN


def test_insufficient_history_returns_empty():
    df = trend_df(n=MIN_BARS - 10)
    assert ichimoku_state(df) == {}


def test_uptrend_is_bullish_regime():
    st = ichimoku_state(trend_df(drift=0.004))
    assert st["regime"] == "bullish"
    assert st["score"] >= 30
    assert st["price_vs_cloud"] == "above"
    assert st["tk_state"] == "bullish"
    assert st["cloud_color"] == "green"


def test_downtrend_is_bearish_regime():
    st = ichimoku_state(trend_df(drift=-0.004))
    assert st["regime"] == "bearish"
    assert st["score"] <= -30
    assert st["price_vs_cloud"] == "below"
    assert st["tk_state"] == "bearish"
    assert st["cloud_color"] == "red"


def test_flat_market_is_neutral():
    st = ichimoku_state(trend_df(drift=0.0, amp=0.004))
    assert st is not None
    # A choppy market must never be a strong regime
    assert abs(st["score"]) < 30
    assert st["regime"] == "neutral"


def test_score_is_bounded_and_symmetric_components():
    up = ichimoku_state(trend_df(drift=0.004))
    down = ichimoku_state(trend_df(drift=-0.004))
    for st in (up, down):
        assert -100 <= st["score"] <= 100
    # Mirror trends should produce mirror scores
    assert up["score"] == pytest.approx(-down["score"])


def test_kijun_distance_sign_matches_trend():
    up = ichimoku_state(trend_df(drift=0.004))
    down = ichimoku_state(trend_df(drift=-0.004))
    assert up["kijun_distance_pct"] > 0
    assert down["kijun_distance_pct"] < 0
