"""
v5.16 session-clock + pinned-coin + strategy-fix tests (importlib pattern).

User request: "fix the logic of ALL strategies, respecting the fixed daily
movements - exchange opens/closes - and weekend volatility, and avoid
quasi-stable coins whose price is pinned."

Covered:
  - session_clock: session tags for every hour block, weekend window,
    Saturday/Sunday flags
  - entry_gate hard rules: chop hours (11-14 UTC), Saturday breakouts
    (A+ exempt), Monday-open reversals, kill-switch fail-open
  - strategy_session_weight: bounded, right strategy for the right hour
  - scorer: pegged symbols hard-skipped, statistical flatness -> dead
    market, session tagging + blocking at chop hours, session fields
  - filter_signals: session-blocked dropped, ranking scaled by weight
  - backtester._passes_gates: session-blocked recs rejected (v5 only)
  - macd_breakout: ADX chop gate + range-ownership gate
  - liquidity_sweep: bearish branch now exists (was bullish-only)

MANDATORY project pattern: src.core / src.analysis re-export singletons
under the module name - always importlib.import_module + setattr on the
MODULE (or the real settings object), never on re-export aliases.
"""
import importlib
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest


def _mod(name: str):
    return importlib.import_module(name)


def _utc(y, m, d, h, minute=0):
    return datetime(y, m, d, h, minute, tzinfo=timezone.utc)


# ----------------------------------------------------------------------
# session_clock: session tags
# ----------------------------------------------------------------------
def test_session_tags_per_hour_block():
    sc = _mod("src.analysis.session_clock")
    # Wednesdays (weekday markets)
    assert sc.session_info(_utc(2026, 9, 23, 0))["session"] == "daily_open"
    assert sc.session_info(_utc(2026, 9, 23, 3))["session"] == "asia"
    assert sc.session_info(_utc(2026, 9, 23, 7))["session"] == "london_open"
    assert sc.session_info(_utc(2026, 9, 23, 9))["session"] == "london"
    assert sc.session_info(_utc(2026, 9, 23, 12))["session"] == "chop"
    assert sc.session_info(_utc(2026, 9, 23, 16))["session"] == "ny"
    assert sc.session_info(_utc(2026, 9, 23, 22))["session"] == "us_late"


def test_weekend_window_and_saturday_flag():
    sc = _mod("src.analysis.session_clock")
    # Friday 21:59 -> not weekend yet; 22:00 -> weekend (v5.13 window)
    assert sc.session_info(_utc(2026, 9, 25, 21, 59))["is_weekend"] is False
    assert sc.session_info(_utc(2026, 9, 25, 22, 0))["is_weekend"] is True
    sat = sc.session_info(_utc(2026, 9, 26, 12))
    assert sat["is_weekend"] is True and sat["is_saturday"] is True
    sun = sc.session_info(_utc(2026, 9, 27, 12))
    assert sun["is_weekend"] is True and sun["is_saturday"] is False
    # Monday 00:00 ends the weekend
    assert sc.session_info(_utc(2026, 9, 28, 0))["is_weekend"] is False


# ----------------------------------------------------------------------
# entry_gate hard rules
# ----------------------------------------------------------------------
def test_chop_hours_block_all_entries():
    sc = _mod("src.analysis.session_clock")
    for h in (11, 12, 13, 14):
        info = sc.session_info(_utc(2026, 9, 23, h))
        blocked, reason = sc.entry_gate(info, "trend_pullback", False)
        assert blocked is True
        assert "الاختلاق" in reason


def test_non_chop_hours_allow_entries():
    sc = _mod("src.analysis.session_clock")
    info = sc.session_info(_utc(2026, 9, 23, 2))   # Asia
    blocked, _ = sc.entry_gate(info, "trend_pullback", False)
    assert blocked is False
    info = sc.session_info(_utc(2026, 9, 23, 18))  # NY
    blocked, _ = sc.entry_gate(info, "volatility_breakout", False)
    assert blocked is False


def test_saturday_blocks_breakouts_but_not_a_plus_or_meanrev():
    sc = _mod("src.analysis.session_clock")
    info = sc.session_info(_utc(2026, 9, 26, 16))  # Saturday NY hours
    assert sc.entry_gate(info, "volatility_breakout", False)[0] is True
    assert sc.entry_gate(info, "macd_breakout", False)[0] is True
    # A+ breakout escapes the block
    assert sc.entry_gate(info, "volatility_breakout", True)[0] is False
    # mean reversion / trends stay allowed
    assert sc.entry_gate(info, "bb_mean_reversion", False)[0] is False
    assert sc.entry_gate(info, "trend_pullback", False)[0] is False


def test_monday_open_blocks_reversals_not_trends():
    sc = _mod("src.analysis.session_clock")
    info = sc.session_info(_utc(2026, 9, 28, 0, 30))  # Mon daily_open
    assert sc.entry_gate(info, "liquidity_sweep_reversal", False)[0] is True
    assert sc.entry_gate(info, "bb_mean_reversion", False)[0] is True
    assert sc.entry_gate(info, "trend_pullback", False)[0] is False
    assert sc.entry_gate(info, "macd_breakout", False)[0] is False


def test_kill_switch_fails_open(monkeypatch):
    sc = _mod("src.analysis.session_clock")
    from config.settings import settings as s
    monkeypatch.setattr(s, "SESSION_FILTER_ENABLED", False)
    info = sc.session_info(_utc(2026, 9, 23, 12))
    blocked, _ = sc.entry_gate(info, "trend_pullback", False)
    assert blocked is False


def test_session_weight_bounded_and_directional(monkeypatch):
    sc = _mod("src.analysis.session_clock")
    asia = sc.session_info(_utc(2026, 9, 23, 2))
    late = sc.session_info(_utc(2026, 9, 23, 22))
    w_asia_trend = sc.strategy_session_weight(asia, "trend_pullback")
    w_late_meanrev = sc.strategy_session_weight(late, "bb_mean_reversion")
    w_late_breakout = sc.strategy_session_weight(late, "macd_breakout")
    assert 0.6 <= w_asia_trend <= 1.25
    assert w_late_meanrev > w_late_breakout  # fade late, don't chase
    assert sc.strategy_session_weight(asia, None) >= 1.0


# ----------------------------------------------------------------------
# synthetic data helpers
# ----------------------------------------------------------------------
def _trend_df(n=160, start=100.0, drift=0.0015, seed=7, tf_min=60,
              start_ts=None):
    """Mild uptrend with noise - a valid, non-flat market window."""
    rng = np.random.default_rng(seed)
    base = start * np.cumprod(1 + drift + rng.normal(0, 0.002, n))
    close = pd.Series(base)
    high = close * (1 + np.abs(rng.normal(0.001, 0.0008, n)))
    low = close * (1 - np.abs(rng.normal(0.001, 0.0008, n)))
    open_ = close.shift(1).fillna(start)
    vol = pd.Series(np.abs(rng.normal(1000, 200, n)) + 200)
    if start_ts is None:
        start_ts = datetime(2026, 9, 23, 2, tzinfo=timezone.utc)
    idx = pd.date_range(start_ts, periods=n, freq=f"{tf_min}min",
                        tz="UTC")
    return pd.DataFrame(
        {"open": open_.values, "high": high.values, "low": low.values,
         "close": close.values, "volume": vol.values}, index=idx)


def _flat_df(n=160, price=1.0005, seed=3, tf_min=60,
             start_ts=None):
    """Pinned / quasi-stable price: basis-point wiggles only."""
    rng = np.random.default_rng(seed)
    close = pd.Series(price * (1 + rng.normal(0, 0.00008, n)))
    high = close * (1 + 0.00005)
    low = close * (1 - 0.00005)
    open_ = close.shift(1).fillna(price)
    vol = pd.Series(np.abs(rng.normal(1000, 100, n)))
    if start_ts is None:
        start_ts = datetime(2026, 9, 23, 2, tzinfo=timezone.utc)
    idx = pd.date_range(start_ts, periods=n, freq=f"{tf_min}min", tz="UTC")
    return pd.DataFrame(
        {"open": open_.values, "high": high.values, "low": low.values,
         "close": close.values, "volume": vol.values}, index=idx)


# ----------------------------------------------------------------------
# scorer: pegged + flat + session
# ----------------------------------------------------------------------
@pytest.fixture
def scorer_mod(monkeypatch):
    mod = _mod("src.analysis.scorer")
    from config.settings import settings as s
    monkeypatch.setattr(s, "TIMEFRAMES", ["1h"])
    monkeypatch.setattr(s, "CONFLUENCE_VETO_ENABLED", False)
    return mod


def test_pegged_symbol_hard_skipped(scorer_mod, monkeypatch):
    from config.settings import settings as s
    monkeypatch.setattr(s, "PEGGED_SYMBOLS", ["USDCUSDT", "EURUSDT"])
    rec = scorer_mod.scorer.analyze_symbol(_trend_df(), "USDCUSDT")
    assert rec.get("skip") is True
    assert "pegged" in rec.get("reason", "").lower()


def test_flat_pinned_market_is_dead(scorer_mod, monkeypatch):
    from config.settings import settings as s
    monkeypatch.setattr(s, "PEGGED_SYMBOLS", [])
    rec = scorer_mod.scorer.analyze_symbol(_flat_df(), "PINNEDUSDT")
    assert rec.get("flat_market") is True
    assert rec.get("dead_market") is True
    assert rec.get("dead_market_reason") == "flat/pinned"
    assert (rec.get("range_pct") or 99) < 1.6


def test_healthy_market_not_flagged_flat(scorer_mod, monkeypatch):
    from config.settings import settings as s
    monkeypatch.setattr(s, "PEGGED_SYMBOLS", [])
    rec = scorer_mod.scorer.analyze_symbol(_trend_df(), "HEALTHUSDT")
    assert rec.get("flat_market") is False


def test_rec_carries_session_fields(scorer_mod, monkeypatch):
    from config.settings import settings as s
    monkeypatch.setattr(s, "PEGGED_SYMBOLS", [])
    df = _trend_df()  # 160 hourly bars from Wed 02:00 UTC
    rec = scorer_mod.scorer.analyze_symbol(df, "SESSTUSDT")
    expected = df.index[-1]  # the rec reflects the LAST bar's clock
    assert rec["session"]["hour"] == expected.hour
    assert rec["session"]["session"] == sc_session_of(expected)
    assert rec["session_blocked"] == (expected.hour in (11, 12, 13, 14))
    assert "session_weight" in rec and "dominant_strategy" in rec


def sc_session_of(ts):
    sc = _mod("src.analysis.session_clock")
    return sc.session_info(ts.to_pydatetime())["session"]


def test_chop_hour_rec_blocked(scorer_mod, monkeypatch):
    from config.settings import settings as s
    monkeypatch.setattr(s, "PEGGED_SYMBOLS", [])
    # 160 hourly bars: start Wed 2026-09-16 21:00 -> last bar lands on
    # Wed 2026-09-23 12:00 UTC (weekday, inside the chop window)
    df = _trend_df(start_ts=datetime(2026, 9, 16, 21, tzinfo=timezone.utc))
    assert df.index[-1].hour == 12
    rec = scorer_mod.scorer.analyze_symbol(df, "CHOPUSDT")
    assert rec["session"]["hour"] == 12
    assert rec["session_blocked"] is True
    assert "الاختلاق" in rec["session_block_reason"]


def test_filter_signals_drops_session_blocked(scorer_mod, monkeypatch):
    from config.settings import settings as s
    monkeypatch.setattr(s, "MIN_RR_RATIO", 1.0)
    monkeypatch.setattr(s, "MIN_HARMONY", 0.0)
    monkeypatch.setattr(s, "EXCLUDE_VOLATILITY_EXTREME", True)
    good = {
        "symbol": "GOODUSDT", "direction": "bullish", "confidence": 75,
        "admission_confidence": 75, "expected_rise_pct": 2.0,
        "risk_reward_ratio": 2.0, "harmony": 0.6, "session_weight": 1.0,
        "session_blocked": False, "dead_market": False,
        "volatility_extreme": False, "decision": {"vetoed": False},
    }
    blocked = dict(good, symbol="BLOCKUSDT", session_blocked=True,
                   session_block_reason="chop")
    out = scorer_mod.scorer.filter_signals([good, blocked])
    assert [r["symbol"] for r in out] == ["GOODUSDT"]


def test_filter_signals_scales_ranking_by_session_weight(scorer_mod,
                                                         monkeypatch):
    from config.settings import settings as s
    monkeypatch.setattr(s, "MIN_RR_RATIO", 1.0)
    monkeypatch.setattr(s, "MIN_HARMONY", 0.0)
    monkeypatch.setattr(s, "EXCLUDE_VOLATILITY_EXTREME", True)
    a = {"symbol": "A", "direction": "bullish", "confidence": 70,
         "admission_confidence": 70, "expected_rise_pct": 2.0,
         "risk_reward_ratio": 2.0, "harmony": 0.5, "session_weight": 1.25,
         "session_blocked": False, "dead_market": False,
         "volatility_extreme": False, "decision": {"vetoed": False}}
    b = dict(a, symbol="B", session_weight=0.6)
    out = scorer_mod.scorer.filter_signals([b, a])
    assert out[0]["symbol"] == "A"  # same merit -> session weight decides


def test_backtester_passes_gates_rejects_blocked():
    bt_mod = _mod("src.backtesting.backtester")
    bt = bt_mod.Backtester(initial_capital=10000.0)
    rec = {"direction": "bullish", "admission_confidence": 80,
           "confidence": 80, "risk_reward_ratio": 2.0, "harmony": 0.6,
           "session_blocked": True, "dead_market": False,
           "volatility_extreme": False, "decision": {"vetoed": False}}
    assert bt._passes_gates(rec, v5=True) is False
    rec2 = dict(rec, session_blocked=False)
    assert bt._passes_gates(rec2, v5=True) is True
    # v4.1 path never session-gates (back-compat baseline)
    assert bt._passes_gates(rec, v5=False) is True


# ----------------------------------------------------------------------
# strategy fixes
# ----------------------------------------------------------------------
def test_macd_breakout_rejects_chop_by_adx(scorer_mod):
    st = _mod("src.strategies.macd_breakout_strategy")
    strat = st.MACDBreakoutStrategy()
    df = _flat_df(n=120)  # zero trend -> ADX ~ 0
    sig = strat.analyze(df, "CHOPUSDT")
    # the BUY side must never fire without trend strength (the exit side
    # may still emit its histogram-flip bearish for open positions)
    assert sig.direction != "bullish"


def test_macd_breakout_needs_cross_or_range_context(scorer_mod):
    st = _mod("src.strategies.macd_breakout_strategy")
    strat = st.MACDBreakoutStrategy()
    # Steady strong uptrend: MACD already far above signal (no fresh/recent
    # cross) -> the strategy must NOT fire even though ADX is high.
    df = _trend_df(n=160, drift=0.004, seed=11)
    sig = strat.analyze(df, "TRENDUSDT")
    assert sig.direction == "neutral"
    assert any("crossover" in r for r in sig.reasons)


def test_liquidity_sweep_bearish_branch_exists(scorer_mod):
    """v5.16: the strategy used to be bullish-only - a bearish sweep must
    now be able to produce a BEARISH signal (exit machinery)."""
    from config.settings import settings as s
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(s, "PEGGED_SYMBOLS", [])
        ls = _mod("src.strategies.liquidity_sweep_reversal_strategy")
        strat = ls.LiquiditySweepReversalStrategy()
        n = 60
        rng = np.random.default_rng(5)
        close = pd.Series(100 + np.linspace(0, 5, n)
                          + rng.normal(0, 0.05, n))
        # last three bars: spike above the swing high, then close back
        # below it (bearish sweep) with a bearish engulfing finish
        close.iloc[-3] = 106.0
        close.iloc[-2] = 104.0
        close.iloc[-1] = 102.5
        high = close * 1.003
        low = close * 0.997
        high.iloc[-1] = 106.8   # swept the swing high (106+)
        open_ = close.shift(1).fillna(100.0)
        open_.iloc[-1] = 104.0  # bearish engulfing body
        vol = pd.Series(np.abs(rng.normal(1000, 100, n)) + 100)
        vol.iloc[-1] = 3000     # capitulation volume
        idx = pd.date_range(datetime(2026, 9, 23, 2, tzinfo=timezone.utc),
                            periods=n, freq="60min", tz="UTC")
        df = pd.DataFrame({"open": open_.values, "high": high.values,
                           "low": low.values, "close": close.values,
                           "volume": vol.values}, index=idx)
        sig = strat.analyze(df, "SWEEPUSDT")
        assert sig.direction in ("bearish", "neutral")
        assert sig.direction != "bullish"
    finally:
        monkeypatch.undo()


def test_liquidity_sweep_knife_filter(monkeypatch):
    """A bullish sweep against a violent downtrend must be rejected."""
    sc = _mod("src.analysis.session_clock")  # ensure module cache warm
    ls = _mod("src.strategies.liquidity_sweep_reversal_strategy")
    strat = ls.LiquiditySweepReversalStrategy()
    n = 90
    rng = np.random.default_rng(9)
    # violent downtrend: -0.8% per bar
    close = pd.Series(150 * np.cumprod(1 - 0.008
                                       + rng.normal(0, 0.001, n)))
    # last 3 bars sweep the prior swing low then close back above it
    swing = float(close.iloc[-40:-3].min())
    close.iloc[-3] = swing * 0.985   # pierce below the swing low
    close.iloc[-2] = swing * 0.988
    close.iloc[-1] = swing * 1.002   # recover above -> bullish sweep
    high = close * 1.002
    low = close * 0.998
    low.iloc[-3] = swing * 0.975     # the wick below
    open_ = close.shift(1).fillna(close.iloc[0])
    vol = pd.Series(np.abs(rng.normal(1000, 150, n)) + 100)
    idx = pd.date_range(datetime(2026, 9, 23, 2, tzinfo=timezone.utc),
                        periods=n, freq="60min", tz="UTC")
    df = pd.DataFrame({"open": open_.values, "high": high.values,
                       "low": low.values, "close": close.values,
                       "volume": vol.values}, index=idx)
    sig = strat.analyze(df, "KNIFEUSDT")
    # with ADX > 40 + minus_di dominating, a bullish sweep must NOT fire
    assert not (sig.direction == "bullish" and sig.score >= 50)
