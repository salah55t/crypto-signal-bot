"""
Tests for the Confluence Nexus (v4) — the integrated decision layer.

Direct evaluate() calls with hand-built layer dicts, plus one
integration test through the scorer to make sure all v4 fields land
in the final recommendation.
"""
import math
import pandas as pd
import pytest

from src.analysis.confluence import ConfluenceEngine, A_PLUS_BONUS
from src.analysis.scorer import scorer


@pytest.fixture()
def engine():
    return ConfluenceEngine()


def _sl_tp(entry=100.0, sl=97.0, tp1=106.0, tp2=112.0,
           label="Limit - Fibonacci 0.618 golden pocket"):
    return {
        "entry_price": entry,
        "stop_loss": sl,
        "take_profit": tp1,
        "take_profit_2": tp2,
        "entry_type": "limit",
        "entry_zone": {"low": 99.0, "high": 100.5},
        "entry_label": label,
        "tp2_label": "Fib 1.272 extension",
        "tp2_rr": 4.0,
        "tp2_distance_pct": 12.0,
    }


def _icho(regime):
    return {"regime": regime, "score": 50.0 if regime == "bullish"
            else (-50.0 if regime == "bearish" else 0.0)}


def _ell(pattern, wave, conf=0.8, projection=None, maturity=None):
    return {"pattern": pattern, "current_wave": wave,
            "wave_confidence": conf, "fib_hits": [],
            "projection": projection, "maturity": maturity,
            "implication": "", "pivots": []}


# ------------------------------------------------------------------
# Ichimoku gate
# ------------------------------------------------------------------

def test_aligned_regime_boosts_confidence(engine):
    out = engine.evaluate("bullish", 70.0, _icho("bullish"), {}, {},
                          _sl_tp(), 100.0)
    assert not out["vetoed"]
    assert out["confidence"] == pytest.approx(70.0 * 1.10)


def test_opposing_regime_vetoes_signal(engine):
    out = engine.evaluate("bullish", 90.0, _icho("bearish"), {}, {},
                          _sl_tp(), 100.0)
    assert out["vetoed"]
    assert "opposes" in out["veto_reason"]


def test_veto_disabled_downgrades_instead(engine):
    out = engine.evaluate("bullish", 90.0, _icho("bearish"), {}, {},
                          _sl_tp(), 100.0, veto_enabled=False)
    assert not out["vetoed"]
    assert out["confidence"] < 90.0


def test_neutral_regime_discounts_confidence(engine):
    out = engine.evaluate("bullish", 70.0, _icho("neutral"), {}, {},
                          _sl_tp(), 100.0)
    assert not out["vetoed"]
    assert out["confidence"] == pytest.approx(70.0 * 0.92)


# ------------------------------------------------------------------
# Elliott timing
# ------------------------------------------------------------------

def test_wave3_bonus(engine):
    out = engine.evaluate("bullish", 70.0, _icho("bullish"),
                          _ell("wave3_up", "3"), {}, _sl_tp(), 100.0)
    assert out["confidence"] == pytest.approx(70.0 * 1.10 + 8.0)


def test_wave4_bonus_and_tp2_promotion(engine):
    sl_tp = _sl_tp(tp2=112.0)
    # 122 extends tp2 (112) and stays inside the 8x SL cap (entry+8*3=124)
    ell = _ell("partial_impulse_up", "4", projection=122.0)
    out = engine.evaluate("bullish", 70.0, _icho("bullish"), ell, {},
                          sl_tp, 100.0)
    assert out["confidence"] == pytest.approx(70.0 * 1.10 + 6.0)
    # projection 122 extends tp2 112 -> promoted
    assert sl_tp["take_profit_2"] == pytest.approx(122.0)
    assert sl_tp["tp2_label"] == "Elliott wave 5 projection"
    assert out["elliott_target"] == pytest.approx(122.0)


def test_tp2_never_shrunk_by_projection(engine):
    sl_tp = _sl_tp(tp2=140.0)
    ell = _ell("partial_impulse_up", "4", projection=122.0)
    engine.evaluate("bullish", 70.0, _icho("bullish"), ell, {}, sl_tp, 100.0)
    assert sl_tp["take_profit_2"] == pytest.approx(140.0)


def test_mature_wave5_penalty(engine):
    ell = _ell("partial_impulse_up", "5", maturity=1.2)
    out = engine.evaluate("bullish", 70.0, _icho("bullish"), ell, {},
                          _sl_tp(), 100.0)
    assert out["confidence"] == pytest.approx(70.0 * 1.10 - 12.0)


def test_completed_impulse_penalty(engine):
    out = engine.evaluate("bullish", 70.0, _icho("bullish"),
                          _ell("impulse_up", "5 (complete)"), {},
                          _sl_tp(), 100.0)
    assert out["confidence"] == pytest.approx(70.0 * 1.10 - 12.0)


def test_counter_cycle_penalty(engine):
    out = engine.evaluate("bullish", 70.0, _icho("bullish"),
                          _ell("impulse_down", "5 (complete)"), {},
                          _sl_tp(), 100.0)
    # completed-impulse penalty AND counter-cycle penalty
    assert out["confidence"] == pytest.approx(70.0 * 1.10 - 12.0 - 10.0)


def test_abc_complete_bonus_for_bullish(engine):
    out = engine.evaluate("bullish", 70.0, _icho("bullish"),
                          _ell("correction_after_up", "C (complete)"), {},
                          _sl_tp(), 100.0)
    assert out["confidence"] == pytest.approx(70.0 * 1.10 + 6.0)


def test_unclear_elliott_no_adjustment(engine):
    out = engine.evaluate("bullish", 70.0, _icho("bullish"),
                          _ell("unclear", None), {}, _sl_tp(), 100.0)
    assert out["confidence"] == pytest.approx(70.0 * 1.10)


# ------------------------------------------------------------------
# A+ setup
# ------------------------------------------------------------------

def test_a_plus_setup_flagged(engine):
    fib = {"golden_zone": {"low": 99.2, "high": 100.4}}
    sl_tp = _sl_tp(label="Limit - Fib 0.618 x support confluence")
    out = engine.evaluate("bullish", 70.0, _icho("bullish"),
                          _ell("wave3_up", "3", conf=0.8), fib, sl_tp, 100.0)
    assert out["a_plus"]
    assert out["confidence"] == pytest.approx(70.0 * 1.10 + 8.0 + A_PLUS_BONUS)


def test_no_a_plus_without_regime_alignment(engine):
    fib = {"golden_zone": {"low": 99.2, "high": 100.4}}
    sl_tp = _sl_tp(label="Limit - Fib 0.618 x support confluence")
    out = engine.evaluate("bullish", 70.0, _icho("neutral"),
                          _ell("wave3_up", "3", conf=0.8), fib, sl_tp, 100.0)
    assert not out["a_plus"]


# ------------------------------------------------------------------
# Confidence bounds + scorer integration
# ------------------------------------------------------------------

def test_confidence_clamped_to_0_100(engine):
    out = engine.evaluate("bullish", 99.0, _icho("bullish"),
                          _ell("wave3_up", "3"), {}, _sl_tp(), 100.0)
    assert 0.0 <= out["confidence"] <= 100.0


def _trend_df(n=120, start=100.0, drift=0.004, amp=0.002):
    rows, price = [], start
    for i in range(n):
        change = drift + amp * math.sin(i * 1.7)
        close = price * (1 + change)
        rows.append({"open": price, "high": max(price, close) * 1.001,
                     "low": min(price, close) * 0.999, "close": close,
                     "volume": 1000.0})
        price = close
    df = pd.DataFrame(rows)
    df.index = pd.date_range("2025-01-01", periods=len(df), freq="h")
    return df


def test_scorer_includes_v4_fields():
    rec = scorer.analyze_symbol(_trend_df(), "TESTUSDT")
    assert "ichimoku" in rec
    assert "elliott" in rec
    assert "decision" in rec
    assert "base_confidence" in rec
    assert 0 <= rec["confidence"] <= 100
    # decision carries the veto info
    assert rec["decision"]["vetoed"] in (True, False)
