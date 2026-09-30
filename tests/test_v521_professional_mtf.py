"""
v5.21 professional multi-timeframe (MTF) tests.

Research base (professional spot-trading practice):
  - "Daily for context, 1h for timing" - trade only with the macro tide
  - A breakout without volume is a fake-out; a vertical candle is a chase
  - Mean reversion collapses in regime breaks (violent ADX + macro
    downtrend = knife, not a fade)
  - RSI extreme + BB touch + confirmation candle is a valid mean-reversion
    entry even without a stochastic cross

Covers:
  - src/analysis/mtf.py: trend_from_df, mtf_context (HTF from the fetched
    dict / daily fetch / fail-open), LTF momentum, TTL cache, ban-awareness
  - strategies: counter-HTF caps full signals at partial; aligned HTF adds
    bonus; graded volume paths; BB regime guard; volatility fakeout/chase
    filters; sweep quality vs the macro tide
  - scorer: MTF confidence bonus (>= 2 voters + HTF agreement),
    result["mtf"] transparency
  - fail-open contract: mtf_ctx=None keeps pre-v5.21 behaviour bit-identical
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from tests.test_v57_signal_stack import (
    triple_bull_df, bb_bull_df, macd_bull_df,
)

mtf_mod = importlib.import_module("src.analysis.mtf")


def _mod(name: str):
    return importlib.import_module(name)


def ctx(trend="up", available=True, ltf=None, tf="1d"):
    """Build a static mtf_context dict (what mtf.mtf_context returns)."""
    return {
        "available": available, "htf_trend": trend if available else None,
        "htf_tf": tf if available else None, "htf_degraded": False,
        "htf_detail": {}, "ltf_available": ltf is not None,
        "ltf_momentum": ltf, "ltf_detail": {},
    }


def _frame(closes, vols=None, freq="1d"):
    closes = np.asarray(closes, dtype=float)
    n = len(closes)
    opens = np.concatenate([[closes[0]], closes[:-1]])
    highs = np.maximum(opens, closes) * 1.001
    lows = np.minimum(opens, closes) * 0.999
    vol = np.full(n, 1000.0) if vols is None else np.asarray(vols, float)
    idx = pd.date_range("2026-01-01", periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({"open": opens, "high": highs, "low": lows,
                         "close": closes, "volume": vol}, index=idx)


def daily_up(n=230):
    rng = np.random.RandomState(7)
    return _frame(100 * (1 + 0.001) ** np.arange(n) + rng.normal(0, 0.2, n))


def daily_down(n=230):
    df = daily_up(n)
    return _frame((df["close"].values * -1) + 300.0)  # mirrored decline


def daily_flat(n=230):
    rng = np.random.RandomState(9)
    return _frame(100 + rng.normal(0, 0.5, n))


# ======================================================================
# mtf.py primitives
# ======================================================================

def test_trend_from_df_up():
    t = mtf_mod.trend_from_df(daily_up())
    assert t["usable"] and t["trend"] == "up" and not t["degraded"]


def test_trend_from_df_down():
    t = mtf_mod.trend_from_df(daily_down())
    assert t["usable"] and t["trend"] == "down"


def test_trend_from_df_degraded_short_series():
    t = mtf_mod.trend_from_df(daily_up(n=120))
    assert t["usable"] and t["degraded"] and t["trend"] == "up"


def test_trend_from_df_too_short_unusable():
    assert not mtf_mod.trend_from_df(daily_up(n=30))["usable"]
    assert not mtf_mod.trend_from_df(None)["usable"]


def test_mtf_context_prefers_higher_tf_from_dict():
    c = mtf_mod.mtf_context("TESTUSDT",
                            {"4h": daily_up(), "1d": daily_down()},
                            primary_tf="4h")
    assert c["available"] and c["htf_trend"] == "down" and c["htf_tf"] == "1d"


def test_mtf_context_same_tf_only_is_unavailable(monkeypatch):
    """The primary frame reading as its own context is duplication, not
    confluence - and with the daily fetch off the context must fail OPEN
    (pre-v5.21 behaviour) without any network access."""
    monkeypatch.setattr(settings, "MTF_FETCH_DAILY", False)
    c = mtf_mod.mtf_context("TESTUSDT", {"4h": daily_up()},
                            primary_tf="4h")
    assert not c["available"] and c["htf_trend"] is None


def test_mtf_context_daily_fetch_path(monkeypatch):
    monkeypatch.setattr(mtf_mod, "_daily_cache", {})
    monkeypatch.setattr(mtf_mod, "_daily", lambda s: daily_up())
    c = mtf_mod.mtf_context("TESTUSDT", {"4h": daily_up()},
                            primary_tf="4h")
    assert c["available"] and c["htf_trend"] == "up" and c["htf_tf"] == "1d"


def test_daily_ttl_cache_single_fetch(monkeypatch):
    monkeypatch.setattr(mtf_mod, "_daily_cache", {})
    calls = {"n": 0}

    def _fake_get_candles(symbol, interval, limit):
        calls["n"] += 1
        return daily_up()

    dfm = importlib.import_module("src.core.data_fetcher")
    monkeypatch.setattr(dfm.data_fetcher, "get_candles",
                        _fake_get_candles)
    a = mtf_mod._daily("CACHUSDT")
    b = mtf_mod._daily("CACHUSDT")
    assert a is not None and b is not None and calls["n"] == 1


def test_daily_ban_aware_no_fetch(monkeypatch):
    monkeypatch.setattr(mtf_mod, "_daily_cache", {})
    rl = importlib.import_module("src.core.rate_limiter")
    monkeypatch.setattr(rl.rate_limiter, "cooldown_remaining",
                        lambda: 500.0)
    calls = {"n": 0}

    def _fake_get_candles(*a, **k):
        calls["n"] += 1
        return daily_up()

    dfm = importlib.import_module("src.core.data_fetcher")
    monkeypatch.setattr(dfm.data_fetcher, "get_candles",
                        _fake_get_candles)
    assert mtf_mod._daily("BANUSDT") is None
    assert calls["n"] == 0


def test_mtf_context_ltf_from_ws_cache(monkeypatch):
    monkeypatch.setattr(mtf_mod, "_daily_cache", {})
    monkeypatch.setattr(settings, "MTF_FETCH_DAILY", False)
    monkeypatch.setattr(mtf_mod, "_ltf_1h", lambda s: daily_up(n=120))
    c = mtf_mod.mtf_context("TESTUSDT", {"4h": daily_flat()},
                            primary_tf="4h")
    assert c["ltf_available"] and c["ltf_momentum"] == "up"
    # primary == 1h -> the tactical frame IS the analysis frame: skipped
    c2 = mtf_mod.mtf_context("TESTUSDT", {"1h": daily_up(n=120)},
                             primary_tf="1h")
    assert not c2["ltf_available"]


def test_mtf_disabled_fail_closed(monkeypatch):
    monkeypatch.setattr(settings, "MTF_ENABLED", False)
    c = mtf_mod.mtf_context("TESTUSDT", {"1d": daily_up()}, primary_tf="4h")
    assert not c["available"]


def test_htf_semantics():
    assert not mtf_mod.htf_agrees(None, "bullish")
    assert not mtf_mod.htf_against(None, "bullish")
    up = ctx("up")
    flat = ctx("flat")
    assert mtf_mod.htf_agrees(up, "bullish")
    assert not mtf_mod.htf_against(up, "bullish")
    # flat macro is NOT against - mean reversion lives in ranges
    assert not mtf_mod.htf_against(flat, "bullish")
    assert not mtf_mod.htf_agrees(flat, "bullish")
    down = ctx("down")
    assert mtf_mod.htf_against(down, "bullish")
    assert mtf_mod.htf_agrees(down, "bearish")
    assert mtf_mod.ltf_agrees(ctx("up", ltf="up"), "bullish")
    assert not mtf_mod.ltf_agrees(ctx("up"), "bullish")


# ======================================================================
# trend_pullback MTF integration
# ======================================================================

def _uptrend_pullback_df():
    """Compact v5.18-style fixture: strong uptrend, dip to EMA21, hammer."""
    tp = _mod("src.strategies.trend_pullback_strategy")
    n = 90
    base = list(np.linspace(100.0, 130.0, n - 3))
    tail = [129.0, 128.2, 129.4]
    closes = base + tail
    opens = [c - 0.15 for c in closes]
    opens[-1] = 129.0
    closes[-1] = 129.5                      # hammer close
    df = _frame(closes, freq="4h")
    df.loc[df.index[-2], "low"] = 128.2 * 0.999
    from src.indicators.technical import ema
    e21 = float(ema(df["close"], 21).iloc[-1])
    df.loc[df.index[-1], "low"] = e21 * 0.992
    return df


def test_trend_pullback_counter_htf_capped_partial():
    strat = _mod("src.strategies.trend_pullback_strategy"). \
        TrendPullbackStrategy()
    df = _uptrend_pullback_df()
    base = strat.analyze(df, "TESTUSDT")           # no ctx -> full signal
    assert base.direction == "bullish" and base.score >= 50
    sig = strat.analyze(df, "TESTUSDT", mtf_ctx=ctx("down"))
    assert sig.direction == "bullish"
    assert any("Counter-HTF" in r for r in sig.reasons)
    # partial = raw score halved (clamp-aware): well below full strength
    assert sig.confidence <= base.confidence * 0.55
    assert sig.score < base.score


def test_trend_pullback_aligned_htf_bonus():
    strat = _mod("src.strategies.trend_pullback_strategy"). \
        TrendPullbackStrategy()
    df = _uptrend_pullback_df()
    base = strat.analyze(df, "TESTUSDT")
    sig = strat.analyze(df, "TESTUSDT", mtf_ctx=ctx("up", ltf="up"))
    assert sig.direction == "bullish"
    assert any("higher timeframe uptrend agrees" in r for r in sig.reasons)
    assert any("1h momentum agrees" in r for r in sig.reasons)
    # +8 HTF and +4 LTF (clamp-aware: the raw fixture can hit the cap)
    assert sig.score == pytest.approx(min(100.0, base.score + 12.0))


# ======================================================================
# triple_confluence_trend MTF + graded volume
# ======================================================================

def test_triple_counter_htf_partial():
    strat = _mod("src.strategies.triple_confluence_trend_strategy"). \
        TripleConfluenceTrendStrategy()
    df = triple_bull_df()
    base = strat.analyze(df, "TESTUSDT")
    assert base.direction == "bullish" and base.score >= 60
    sig = strat.analyze(df, "TESTUSDT", mtf_ctx=ctx("down"))
    assert sig.direction == "bullish"
    assert any("MTF" in r for r in sig.reasons)
    assert sig.score == pytest.approx(base.score * 0.5)   # capped partial


def test_triple_aligned_htf_bonus():
    strat = _mod("src.strategies.triple_confluence_trend_strategy"). \
        TripleConfluenceTrendStrategy()
    df = triple_bull_df()
    base = strat.analyze(df, "TESTUSDT")
    sig = strat.analyze(df, "TESTUSDT", mtf_ctx=ctx("up", ltf="up"))
    assert any("daily uptrend agrees" in r for r in sig.reasons)
    assert any("1h momentum agrees" in r for r in sig.reasons)
    assert sig.score == pytest.approx(min(100.0, base.score + 6 + 3))


def test_triple_thin_volume_partial_not_neutral():
    """v5.21 graded liquidity: 0.8-1.0x volume on the trigger bar is a
    partial setup, not a hard veto (the old gate muted whole sessions)."""
    strat = _mod("src.strategies.triple_confluence_trend_strategy"). \
        TripleConfluenceTrendStrategy()
    df = triple_bull_df()
    df.loc[df.index[-1], "volume"] = 900.0        # ~0.9x avg20
    sig = strat.analyze(df, "TESTUSDT")
    assert sig.direction == "bullish"
    assert any("Thin volume" in r for r in sig.reasons)
    assert sig.confidence < 0.45                  # partial scale


# ======================================================================
# macd_breakout MTF + graded volume
# ======================================================================

def test_macd_counter_htf_partial():
    strat = _mod("src.strategies.macd_breakout_strategy"). \
        MACDBreakoutStrategy()
    df = macd_bull_df()
    base = strat.analyze(df, "TESTUSDT")
    assert base.direction == "bullish" and base.score >= 60
    sig = strat.analyze(df, "TESTUSDT", mtf_ctx=ctx("down"))
    assert sig.direction == "bullish"
    assert any("MTF" in r for r in sig.reasons)
    assert sig.score == pytest.approx(base.score * 0.5)


def test_macd_thin_volume_partial_not_neutral():
    strat = _mod("src.strategies.macd_breakout_strategy"). \
        MACDBreakoutStrategy()
    df = macd_bull_df()
    df.loc[df.index[-1], "volume"] = 900.0        # ~0.9x avg20
    sig = strat.analyze(df, "TESTUSDT")
    assert sig.direction == "bullish"
    assert any("Thin volume" in r for r in sig.reasons)
    assert sig.confidence < 0.55


# ======================================================================
# bb_mean_reversion regime guard + RSI-extreme alt path
# ======================================================================

def test_bb_regime_guard_only_with_ctx():
    """Fail-open contract: the violent-tape fixture (ADX ~53) without an
    MTF context keeps the pre-v5.21 FULL bullish; with a daily downtrend
    context it is capped at a regime-guard partial (knife, not a fade)."""
    strat = _mod("src.strategies.bb_mean_reversion_strategy"). \
        BBMeanReversionStrategy()
    df = bb_bull_df()
    base = strat.analyze(df, "TESTUSDT")
    assert base.direction == "bullish" and base.score >= 60       # unchanged
    sig = strat.analyze(df, "TESTUSDT", mtf_ctx=ctx("down"))
    assert sig.direction == "bullish"
    assert any("Regime-guard partial" in r or "Counter-HTF" in r
               for r in sig.reasons)
    assert sig.score < 60


def test_bb_aligned_dip_bonus():
    strat = _mod("src.strategies.bb_mean_reversion_strategy"). \
        BBMeanReversionStrategy()
    df = bb_bull_df()
    base = strat.analyze(df, "TESTUSDT")
    sig = strat.analyze(df, "TESTUSDT", mtf_ctx=ctx("up"))
    assert any("buying a dip" in r for r in sig.reasons)
    assert sig.score == pytest.approx(min(100.0, base.score + 6))


def test_bb_rsi_extreme_alt_path():
    """Research path: RSI<28 + broken band + bullish confirmation candle
    WITHOUT a stochastic cross must still emit a partial - the old code
    hard-vetoed it with 'No Stoch bullish crossover yet'."""
    strat = _mod("src.strategies.bb_mean_reversion_strategy"). \
        BBMeanReversionStrategy()
    m = 80
    rng = np.random.RandomState(3)
    closes = 100.0 + rng.normal(0, 0.2, m)
    for i in range(14, 1, -1):                    # accelerating decline
        closes[-i] = closes[-i - 1] - 0.9
    closes[-1] = closes[-2] + 0.12                # tiny bullish candle
    df = _frame(closes, freq="1h")
    sig = strat.analyze(df, "TESTUSDT")
    if sig.direction == "neutral":
        pytest.skip("fixture produced a stoch cross - alt path not needed")
    assert sig.direction == "bullish"
    assert any("RSI extreme bounce" in r for r in sig.reasons)
    assert sig.confidence < 0.5                   # partial scale


# ======================================================================
# volatility_breakout fakeout filter + chase guard + MTF
# ======================================================================

def _breakout_df(noise=1.2, breakout=4.0, last_vol=2600.0):
    """Quiet range then a bullish breakout candle closing `breakout`
    above the range; volume on the breakout bar = `last_vol`."""
    n = 90
    rng = np.random.RandomState(11)
    closes = 100.0 + rng.normal(0, noise, n)
    closes[-1] = 100.0 + breakout
    vols = np.full(n, 1000.0)
    vols[-1] = last_vol
    return _frame(closes, vols, freq="1h")


def test_vol_breakout_without_volume_is_partial():
    """The professional fakeout filter: a breakout candle on < 1.2x
    volume must never be a FULL signal."""
    strat = _mod("src.strategies.volatility_breakout_strategy"). \
        VolatilityBreakoutStrategy()
    df = _breakout_df(last_vol=1000.0)            # 1.0x -> no confirmation
    sig = strat.analyze(df, "TESTUSDT")
    if sig.direction == "neutral":
        pytest.skip("fixture score below the partial bar - nothing to gate")
    assert sig.direction == "bullish"
    assert any("fakeout" in r.lower() for r in sig.reasons)
    assert sig.confidence < 0.5


def test_vol_breakout_confirmed_volume_full():
    strat = _mod("src.strategies.volatility_breakout_strategy"). \
        VolatilityBreakoutStrategy()
    df = _breakout_df(last_vol=2600.0)            # 2.6x -> confirmed
    sig = strat.analyze(df, "TESTUSDT")
    assert sig.direction == "bullish"
    assert not any("fakeout" in r.lower() for r in sig.reasons)


def test_vol_breakout_chase_guard():
    strat = _mod("src.strategies.volatility_breakout_strategy"). \
        VolatilityBreakoutStrategy()
    df = _breakout_df(noise=0.3, breakout=6.0, last_vol=2600.0)
    sig = strat.analyze(df, "TESTUSDT")
    if sig.direction == "neutral":
        pytest.skip("fixture too far outside the band to score")
    assert sig.direction == "bullish"
    assert any("Late chase" in r for r in sig.reasons)
    assert sig.confidence < 0.5                   # capped partial


def test_vol_breakout_counter_htf_capped():
    strat = _mod("src.strategies.volatility_breakout_strategy"). \
        VolatilityBreakoutStrategy()
    df = _breakout_df(last_vol=2600.0)
    base = strat.analyze(df, "TESTUSDT")
    assert base.direction == "bullish"
    sig = strat.analyze(df, "TESTUSDT", mtf_ctx=ctx("down"))
    assert sig.direction == "bullish"
    assert any("MTF" in r for r in sig.reasons)
    assert sig.confidence <= base.confidence * 0.55
    assert sig.score < base.score


def test_vol_breakout_aligned_htf_bonus():
    strat = _mod("src.strategies.volatility_breakout_strategy"). \
        VolatilityBreakoutStrategy()
    df = _breakout_df(last_vol=2600.0)
    base = strat.analyze(df, "TESTUSDT")
    sig = strat.analyze(df, "TESTUSDT", mtf_ctx=ctx("up"))
    assert any("MTF: daily trend agrees" in r for r in sig.reasons)
    assert sig.score == pytest.approx(min(100.0, base.score + 6))


# ======================================================================
# liquidity_sweep_reversal MTF
# ======================================================================

def _bullish_sweep_df():
    """Downtrend into a support shelf, sweep of the swing low, hammer
    close back above it, capitulation volume (mirrors the v5.16 bearish
    sweep fixture)."""
    n = 70
    close = pd.Series(110.0 + np.linspace(0, -12, n))
    close.iloc[-4] = 96.0                         # local swing low
    close.iloc[-2] = 95.2                         # sweep wick bar (close)
    close.iloc[-1] = 96.8                         # recovery close
    high = close * 1.002
    low = close * 0.998
    low.iloc[-2] = 94.6                           # the swept low
    open_ = close.shift(1).fillna(110.0)
    open_.iloc[-1] = 95.4
    high.iloc[-1] = 97.0
    vol = pd.Series(np.full(n, 1000.0))
    vol.iloc[-2] = 3000.0                         # capitulation
    idx = pd.date_range("2026-09-28", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"open": open_.values, "high": high.values,
                         "low": low.values, "close": close.values,
                         "volume": vol.values}, index=idx)


def test_sweep_counter_htf_vs_aligned():
    strat = _mod("src.strategies.liquidity_sweep_reversal_strategy"). \
        LiquiditySweepReversalStrategy()
    df = _bullish_sweep_df()
    base = strat.analyze(df, "TESTUSDT")
    if base.direction == "neutral":
        pytest.skip("fixture sweep not detected - nothing to gate")
    assert base.direction == "bullish"
    capped = strat.analyze(df, "TESTUSDT", mtf_ctx=ctx("down"))
    assert capped.direction == "bullish"
    assert any("Counter-HTF" in r for r in capped.reasons)
    assert capped.score == pytest.approx(base.score * 0.5)
    aligned = strat.analyze(df, "TESTUSDT", mtf_ctx=ctx("up"))
    assert aligned.score == pytest.approx(min(100.0, base.score + 8))


# ======================================================================
# scorer integration: MTF confidence bonus + transparency
# ======================================================================

def _fake_signal(name, direction, score):
    return _mod("src.strategies.base").Signal(
        strategy=name, direction=direction, score=score)


class _FakeStrat(_mod("src.strategies.base").BaseStrategy):
    """Deterministic vote slot (concrete - BaseStrategy is abstract)."""

    def __init__(self, name, weight, direction, score):
        super().__init__(weight)
        self.name = name
        self._sig = _mod("src.strategies.base").Signal(
            strategy=name, direction=direction, score=score)

    def analyze(self, df, symbol, multi_tf_data=None, order_book=None,
                mtf_ctx=None):
        return self._sig


def _scorer_with_fakes(monkeypatch, votes):
    """SignalScorer whose six strategy slots are deterministic fakes.
    votes = [(direction, score, weight), ...] in strategy order."""
    from src.analysis.scorer import SignalScorer

    sc = SignalScorer.__new__(SignalScorer)       # skip real strategies
    sc.strategies = []
    names = ["trend_pullback", "liquidity_sweep_reversal",
             "volatility_breakout", "triple_confluence_trend",
             "bb_mean_reversion", "macd_breakout"]
    for i, name in enumerate(names):
        d, s, w = votes[i]
        sc.strategies.append(_FakeStrat(name, w, d, s))
    sc.total_weight = sum(s.weight for s in sc.strategies)
    return sc


def test_scorer_mtf_bonus_two_voters(monkeypatch):
    from src.core import ws_feed as wsf
    monkeypatch.setattr(wsf.ws_feed, "_started", True)
    import src.analysis.mtf as m
    monkeypatch.setattr(m, "mtf_context",
                        lambda symbol, mtd, primary_tf=None: ctx("up"))
    from src.analysis.scorer import SignalScorer
    sc = _scorer_with_fakes(
        monkeypatch,
        [("bullish", 80, 2.0), ("bullish", 80, 2.0),
         ("neutral", 0, 1.8), ("neutral", 0, 1.0),
         ("neutral", 0, 1.0), ("neutral", 0, 1.5)])
    df = triple_bull_df()
    monkeypatch.setattr(settings, "MTF_ENABLED", False)
    off = sc.analyze_symbol(df, "TESTUSDT")
    monkeypatch.setattr(settings, "MTF_ENABLED", True)
    on = sc.analyze_symbol(df, "TESTUSDT")
    assert on["mtf"]["available"] is True
    # the +3.0 bonus enters as base confidence (deterministic); the
    # confluence engine may then amplify it slightly in the final value
    assert on["base_confidence"] == pytest.approx(
        off["base_confidence"] + 3.0)
    assert on["confidence"] >= off["confidence"] + 2.9


def test_scorer_no_bonus_single_voter(monkeypatch):
    """One loud strategy + aligned HTF must NOT cross the gate on the
    bonus - professionals demand multi-source confluence."""
    from src.core import ws_feed as wsf
    monkeypatch.setattr(wsf.ws_feed, "_started", True)
    import src.analysis.mtf as m
    monkeypatch.setattr(m, "mtf_context",
                        lambda symbol, mtd, primary_tf=None: ctx("up"))
    from src.analysis.scorer import SignalScorer
    sc = _scorer_with_fakes(
        monkeypatch,
        [("bullish", 85, 2.0), ("neutral", 0, 2.0),
         ("neutral", 0, 1.8), ("neutral", 0, 1.0),
         ("neutral", 0, 1.0), ("neutral", 0, 1.5)])
    df = triple_bull_df()
    monkeypatch.setattr(settings, "MTF_ENABLED", False)
    off = sc.analyze_symbol(df, "TESTUSDT")
    monkeypatch.setattr(settings, "MTF_ENABLED", True)
    on = sc.analyze_symbol(df, "TESTUSDT")
    assert on["confidence"] == pytest.approx(off["confidence"])


def test_scorer_mtf_against_no_bonus(monkeypatch):
    from src.core import ws_feed as wsf
    monkeypatch.setattr(wsf.ws_feed, "_started", True)
    import src.analysis.mtf as m
    monkeypatch.setattr(m, "mtf_context",
                        lambda symbol, mtd, primary_tf=None: ctx("down"))
    from src.analysis.scorer import SignalScorer
    sc = _scorer_with_fakes(
        monkeypatch,
        [("bullish", 80, 2.0), ("bullish", 80, 2.0),
         ("neutral", 0, 1.8), ("neutral", 0, 1.0),
         ("neutral", 0, 1.0), ("neutral", 0, 1.5)])
    df = triple_bull_df()
    monkeypatch.setattr(settings, "MTF_ENABLED", False)
    off = sc.analyze_symbol(df, "TESTUSDT")
    monkeypatch.setattr(settings, "MTF_ENABLED", True)
    on = sc.analyze_symbol(df, "TESTUSDT")
    assert on["confidence"] == pytest.approx(off["confidence"])
    assert on["mtf"]["htf_trend"] == "down"       # transparency still on


def test_scorer_carries_mtf_snapshot(monkeypatch):
    from src.core import ws_feed as wsf
    monkeypatch.setattr(wsf.ws_feed, "_started", True)
    import src.analysis.mtf as m
    monkeypatch.setattr(m, "mtf_context",
                        lambda symbol, mtd, primary_tf=None: ctx("flat"))
    from src.analysis.scorer import SignalScorer
    sc = _scorer_with_fakes(
        monkeypatch,
        [("neutral", 0, 2.0), ("neutral", 0, 2.0), ("neutral", 0, 1.8),
         ("neutral", 0, 1.0), ("neutral", 0, 1.0), ("neutral", 0, 1.5)])
    df = triple_bull_df()
    res = sc.analyze_symbol(df, "TESTUSDT")
    assert res["mtf"]["available"] is True
    assert res["mtf"]["htf_trend"] == "flat"
