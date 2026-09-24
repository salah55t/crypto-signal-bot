"""
v5.7 Signal Stack - Unit Tests

User specification: program three strategies and add them to the code:
  1. TripleConfluenceTrendStrategy - EMA200/EMA50 (direction) + RSI 14
     (momentum) + Volume/OBV (liquidity). Exit: EMA50 break, or RSI>70 +
     bearish divergence.
  2. BBMeanReversionStrategy - BB(20,2) (volatility) + Stochastic (14,3,3)
     (momentum). Buy at lower band + oversold + fresh K/D cross in the
     oversold zone; exit at upper band + RSI>70 (TP1 = middle band).
  3. MACDBreakoutStrategy - EMA50 (direction) + MACD(12,26,9) (momentum) +
     Volume (liquidity). Zero-line cross + histogram flip; laddered
     multi-TP + trailing SL map onto the bot's TP1/TP2 + chandelier.

Golden rule ("Signal Stack Framework"): one indicator per class inside
each strategy - never same-class stacking.

Infrastructure covered:
  - Settings-gated registration in SignalScorer (6 strategies)
  - Confluence scale preservation: confluence is measured against the
    calibrated reference weight (5.8), NOT the live total - adding
    strategies must not silently raise the 3-of-6 agreement bar.
  - CANDLE_LIMIT 200 -> 300 (EMA200 warmup) + WS MAX_BARS 400
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from src.strategies import (
    TrendPullbackStrategy,
    LiquiditySweepReversalStrategy,
    VolatilityBreakoutStrategy,
    TripleConfluenceTrendStrategy,
    BBMeanReversionStrategy,
    MACDBreakoutStrategy,
)
from src.strategies.base import Signal
from src.analysis.scorer import SignalScorer


# ============================================
# Synthetic fixtures (calibrated, seed=7)
# ============================================

def mk_df(closes, vols=None):
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    vols = np.asarray(vols if vols is not None else [1000.0] * n, dtype=float)
    return pd.DataFrame({
        "open": closes * 0.999, "high": closes * 1.002,
        "low": closes * 0.997, "close": closes, "volume": vols,
    })


def triple_bull_df():
    """Uptrend -> 8-bar pullback -> 3-bar push with volume. Triggers the
    full checklist: price > EMA200, EMA50 > EMA200, RSI fresh cross > 50
    (< 70), volume > avg20."""
    n = 225
    rng = np.random.RandomState(7)
    closes = 100 * (1 + 0.0005) ** np.arange(n) + rng.normal(0, 0.12, n)
    for i in range(8, 0, -1):
        closes[-i - 3] = closes[-i - 4] - 0.25
    for j in range(3):
        closes[-3 + j] = closes[-4 + j] + 0.75
    vols = np.full(n, 1000.0)
    vols[-1] = 1600
    return mk_df(closes, vols)


def triple_bear_df():
    """Same uptrend, then a 2-bar crash below EMA50 -> EMA50-breakdown exit."""
    df = triple_bull_df()
    closes = df["close"].copy()
    closes.iloc[-2] = closes.iloc[-3] * 0.978
    closes.iloc[-1] = closes.iloc[-2] * 0.975
    return mk_df(closes.values)


def bb_bull_df():
    """Range -> accelerating decline -> capitulation dip -> small bullish
    bounce. Lower band touched, Stoch(14,3,3) fresh cross up in oversold."""
    m = 110
    rng = np.random.RandomState(7)
    closes = 100.0 + rng.normal(0, 0.25, m)
    steps = np.linspace(0.15, 1.1, 18)
    idx = 0
    for i in range(19, 2, -1):
        closes[-i] = closes[-i - 1] - steps[idx]
        idx += 1
    closes[-2] = closes[-3] - 2.5   # capitulation dip (band break)
    closes[-1] = closes[-2] + 0.35  # small bullish bounce
    return mk_df(closes)


def bb_bear_df():
    """Strong rally with a mid-series dip (so RSI is finite), then a final
    +5% pierce ABOVE the upper band with RSI > 70 -> full exit."""
    n = 90
    closes = np.full(n, 100.0)
    for i in range(1, n):
        closes[i] = closes[i - 1] * 1.012
    closes[40] = closes[39] * 0.995  # guaranteed down bar (RSI sanity)
    closes[-1] = closes[-2] * 1.05   # pierce the upper band
    return mk_df(closes)


def macd_bull_df():
    """Flat base -> small dip -> 3-bar burst -> fresh MACD cross + histogram
    flip + price above EMA50 + volume."""
    k = 120
    rng = np.random.RandomState(7)
    closes = 100.0 + rng.normal(0, 0.02, k)
    closes[-6] = closes[-7] - 0.15
    closes[-5] = closes[-6] - 0.15
    closes[-4] = closes[-5] - 0.1
    for j in range(1, 4):
        closes[-j] = closes[-j - 1] * 1.022
    vols = np.full(k, 1000.0)
    vols[-1] = 1800
    return mk_df(closes, vols)


def macd_bear_df():
    """Two-wave advance, then a 7-bar decay that flips the histogram
    negative exactly on the last bar -> MACD-bent-down exit."""
    k = 90
    rng = np.random.RandomState(7)
    closes = 100.0 + rng.normal(0, 0.05, k)
    for j in range(4):
        closes[68 + j] = closes[67 + j] * 1.015   # burst 1
    for j in range(4):
        closes[73 + j] = closes[72 + j] * 0.9995  # cool-off
    for j in range(4):
        closes[78 + j] = closes[77 + j] * 1.012   # burst 2
    closes[82] = closes[81]
    for j in range(7):
        closes[83 + j] = closes[82 + j] * 0.997   # decay -> flip
    return mk_df(closes)


# ============================================
# Settings + registration
# ============================================

def test_settings_v57_defaults():
    assert settings.STRATEGY_TRIPLE_TREND_ENABLED is True
    assert settings.STRATEGY_TRIPLE_TREND_WEIGHT == 1.6
    assert settings.STRATEGY_BB_MEAN_REV_ENABLED is True
    assert settings.STRATEGY_BB_MEAN_REV_WEIGHT == 1.2
    assert settings.STRATEGY_MACD_BREAKOUT_ENABLED is True
    assert settings.STRATEGY_MACD_BREAKOUT_WEIGHT == 1.4
    assert settings.STRATEGY_CONFLUENCE_REF_WEIGHT == 5.8
    assert settings.CANDLE_LIMIT == 300


def test_scorer_registers_six_strategies():
    scorer = SignalScorer()
    names = [s.name for s in scorer.strategies]
    assert names == [
        "trend_pullback", "liquidity_sweep_reversal", "volatility_breakout",
        "triple_confluence_trend", "bb_mean_reversion", "macd_breakout",
    ]
    # 2.0 + 2.0 + 1.8 + 1.6 + 1.2 + 1.4 = 10.0
    assert scorer.total_weight == pytest.approx(10.0)


def test_scorer_respects_toggles(monkeypatch):
    monkeypatch.setattr(settings, "STRATEGY_TRIPLE_TREND_ENABLED", False)
    scorer = SignalScorer()
    names = [s.name for s in scorer.strategies]
    assert "triple_confluence_trend" not in names
    assert len(names) == 5  # original 3 + the two remaining v5.7 ones


# ============================================
# Confluence scale preservation (no dilution)
# ============================================

class _Strat:
    def __init__(self, weight):
        self.weight = weight


def _bull(score):
    return Signal(strategy="x", direction="bullish", score=score)


def test_lone_signal_confidence_unchanged_on_original_stack():
    """One strong strategy @80 on the ORIGINAL 3-stack -> ~64% (calibration)."""
    scorer = SignalScorer()
    old3 = [TrendPullbackStrategy(2.0), LiquiditySweepReversalStrategy(2.0),
            VolatilityBreakoutStrategy(1.8)]
    v = scorer._compute_confidence([_bull(80), Signal(strategy="y"),
                                    Signal(strategy="z")], old3)
    assert v["confidence"] == pytest.approx(64.07, abs=0.05)


def test_lone_signal_confidence_not_diluted_on_six_stack():
    """The SAME lone signal keeps ~64% even with 6 registered strategies -
    adding strategies must never silently raise the admission bar."""
    scorer = SignalScorer()
    six = scorer.strategies
    signals = [_bull(80)] + [Signal(strategy="y") for _ in range(5)]
    v = scorer._compute_confidence(signals, six)
    assert v["confidence"] == pytest.approx(64.07, abs=0.05)


def test_confluence_capped_and_additive():
    """All six agreeing -> confluence capped at 1.0; agreements from the new
    trio ADD to a two-strategy signal instead of diluting it."""
    scorer = SignalScorer()
    six = scorer.strategies
    # all agree @80
    v_all = scorer._compute_confidence([_bull(80)] * 6, six)
    assert v_all["confluence"] == 1.0
    assert v_all["confidence"] == pytest.approx(87.0, abs=0.05)
    # original two agree (4.0) + macd breakout (1.4) -> confluence 5.4/5.8 (was 4/5.8)
    sigs = [_bull(80), _bull(80), Signal(strategy="z"), Signal(strategy="z"),
            Signal(strategy="z"), _bull(80)]
    v = scorer._compute_confidence(sigs, six)
    assert v["confluence"] == pytest.approx(5.4 / 5.8, abs=1e-6)


# ============================================
# Strategy 1: Triple Confluence Trend
# ============================================

def test_triple_trend_bullish():
    sig = TripleConfluenceTrendStrategy().analyze(triple_bull_df(), "TESTUSDT")
    assert sig.direction == "bullish"
    assert sig.score >= 60
    reasons = " | ".join(sig.reasons)
    assert "EMA200" in reasons and "RSI" in reasons
    assert sig.details["stack"]["liquidity"] == "Volume/OBV"


def test_triple_trend_bearish_on_ema50_break():
    sig = TripleConfluenceTrendStrategy().analyze(triple_bear_df(), "TESTUSDT")
    assert sig.direction == "bearish"
    assert sig.score <= -55
    assert "EMA50 breakdown" in " ".join(sig.reasons)


def test_triple_trend_needs_ema200_history():
    sig = TripleConfluenceTrendStrategy().analyze(
        triple_bull_df().iloc[-150:].reset_index(drop=True), "TESTUSDT")
    assert sig.direction == "neutral"
    assert "Insufficient data" in sig.reasons[0]


# ============================================
# Strategy 2: BB Mean Reversion
# ============================================

def test_bb_mean_rev_bullish():
    sig = BBMeanReversionStrategy().analyze(bb_bull_df(), "TESTUSDT")
    assert sig.direction == "bullish"
    assert sig.score >= 60
    reasons = " | ".join(sig.reasons)
    assert "Lower BB" in reasons
    assert "Stoch cross" in reasons
    # user-spec exit ladder carried in details
    assert sig.details["tp1_middle_band"] > 0
    assert sig.details["tp2_upper_band"] > sig.details["tp1_middle_band"]


def test_bb_mean_rev_bearish_at_upper_band_with_rsi_overbought():
    sig = BBMeanReversionStrategy().analyze(bb_bear_df(), "TESTUSDT")
    assert sig.direction == "bearish"
    assert sig.score <= -55
    assert "mean reversion complete" in " ".join(sig.reasons)


def test_bb_mean_rev_neutral_mid_range():
    rng = np.random.RandomState(3)
    closes = 100 + rng.normal(0, 0.3, 80)  # range-bound: no band touch
    sig = BBMeanReversionStrategy().analyze(mk_df(closes), "TESTUSDT")
    assert sig.direction == "neutral"


def test_bb_slow_stochastic_uses_14_3_3():
    """(14,3,3): raw %K additionally smoothed by 3, %D = SMA(slow K, 3)."""
    from src.indicators.technical import stochastic, sma
    df = bb_bull_df()
    raw = stochastic(df["high"], df["low"], df["close"], 14, 3)["k"]
    slow_k = sma(raw, 3)
    slow_d = sma(slow_k, 3)
    # smoothing must actually change the latest value vs the raw %K
    assert abs(float(slow_k.iloc[-1]) - float(raw.iloc[-1])) > 0
    # %D is the 3-bar mean of slow %K
    assert slow_d.iloc[-1] == pytest.approx(
        float(slow_k.iloc[-3:].mean()), abs=1e-6)


# ============================================
# Strategy 3: MACD Breakout
# ============================================

def test_macd_breakout_bullish():
    sig = MACDBreakoutStrategy().analyze(macd_bull_df(), "TESTUSDT")
    assert sig.direction == "bullish"
    assert sig.score >= 60
    reasons = " | ".join(sig.reasons)
    assert "MACD cross" in reasons
    assert "Histogram" in reasons


def test_macd_breakout_bearish_on_histogram_flip():
    sig = MACDBreakoutStrategy().analyze(macd_bear_df(), "TESTUSDT")
    assert sig.direction == "bearish"
    assert sig.score <= -55


def test_macd_breakout_needs_price_above_ema50():
    """A SLOW drift below EMA50 (no fresh cross, no hist flip) must reach
    the direction gate and return neutral 'below EMA50'. A sudden crash
    would legitimately fire the bearish histogram-flip exit instead."""
    closes = macd_bear_df()["close"].values.tolist()
    for _ in range(15):  # gentle appended drift: hist stays negative,
        closes.append(closes[-1] * 0.999)  # no new cross, price sinks < EMA50
    sig = MACDBreakoutStrategy().analyze(mk_df(closes), "TESTUSDT")
    assert sig.direction == "neutral"
    assert "below EMA50" in sig.reasons[0]


# ============================================
# Integration: through the full scorer pipeline
# ============================================

def test_scorer_analyze_symbol_runs_all_six():
    from src.indicators.technical import atr
    scorer = SignalScorer()
    df = triple_bull_df()
    result = scorer.analyze_symbol(df, "TESTUSDT")
    sig_names = [s["strategy"] for s in result["signals"]]
    assert len(sig_names) == 6
    assert "triple_confluence_trend" in sig_names
    assert result["direction"] in ("bullish", "bearish", "neutral")
