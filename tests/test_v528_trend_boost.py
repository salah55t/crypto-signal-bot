"""v5.28: evidence-based strategy overhaul tests.

Research context (scripts/research/): 39 top-volume USDT pairs x ~500 days
of real 4h klines, honest 0.24% round-trip costs, the production exit
lifecycle, 4 time-folds and bootstrap CIs selected:

  NEW trend boost channel (4h, WS-cached, zero REST weight):
    - donchian_break: +28.3 bps/trade, PF 1.98, all folds positive
    - st_flip:        +14.7 bps/trade, PF 1.50, all folds positive
  RETIRED composites (negative/breakeven edge):
    - liquidity_sweep_reversal -6.9 bps, PF 0.81, ALL folds negative
    - trend_pullback           -0.5 bps, PF 0.99, folds mixed
  STARVATION FIX: STRATEGY_CONFLUENCE_REF_WEIGHT 5.8 -> 2.0 (solo votes
  needed strength >= 86-93 to clear MIN_CONFIDENCE=68; live evidence:
  ZERO composite trades ever opened).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
import src.analysis.trend_scanner as trend_scanner_module
from src.analysis.trend_scanner import TrendScanner
from src.analysis.analyzer import build_trend_rec
from src.indicators.supertrend import supertrend


# ============================================================
# Fixtures
# ============================================================

def _base_frame(n: int = 300, start: float = 100.0) -> pd.DataFrame:
    """Sideways 4h frame, ATR% ~1% (above the 0.30% dead floor)."""
    rng = np.random.default_rng(42)
    close = start + np.cumsum(rng.normal(0, start * 0.004, n))
    close = np.clip(close, start * 0.85, start * 1.15)
    high = close * (1 + 0.006)
    low = close * (1 - 0.006)
    open_ = np.r_[close[0], close[:-1]]
    vol = np.full(n, 1000.0)
    ts = pd.date_range("2026-01-01", periods=n, freq="4h", tz="UTC")
    return pd.DataFrame({
        "open_time": ts, "open": open_, "high": high, "low": low,
        "close": close, "volume": vol, "close_time": ts,
        "quote_volume": vol * close, "trades": 100,
    })


def _breakout_frame() -> pd.DataFrame:
    """Sideways frame whose LAST CLOSED bar crosses the prior 55-bar high
    with 2x volume."""
    df = _base_frame()
    c = df["close"]
    # flatten the last 60 bars so the channel top is known and beaten
    c.iloc[-60:] = float(c.iloc[-70])
    df["high"].iloc[-60:] = c.iloc[-60:] * 1.002
    df["low"].iloc[-60:] = c.iloc[-60:] * 0.998
    df["open"].iloc[-60:] = c.iloc[-60:]
    # final bar: strong close above the prior 55-bar high, 2x volume
    prev = float(c.iloc[-2])
    df.iloc[-1, df.columns.get_loc("open")] = prev
    df.iloc[-1, df.columns.get_loc("close")] = prev * 1.03
    df.iloc[-1, df.columns.get_loc("high")] = prev * 1.031
    df.iloc[-1, df.columns.get_loc("low")] = prev * 0.999
    df.iloc[-1, df.columns.get_loc("volume")] = 2000.0
    return df


def _flip_frame() -> pd.DataFrame:
    """Uptrend -> steep pullback (ST flips down) -> recovery ramp where the
    PRODUCTION SuperTrend(10,3) flips back up above EMA200. The frame is
    trimmed so the LAST closed bar IS the flip bar. Parameters are picked
    from a small deterministic grid (first combo satisfying both the flip
    and the EMA200 regime condition)."""
    for depth, ramp in [(0.96, 1.10), (0.95, 1.10), (0.94, 1.12),
                        (0.933, 1.08), (0.93, 1.15), (0.92, 1.15)]:
        df = _base_frame(300)
        c = df["close"].values.astype(float)
        c[:] = 100.0
        c[:250] = 100.0 + np.linspace(0, 5, 250)
        c[250:270] = np.linspace(c[249], c[249] * depth, 20)
        c[270:] = np.linspace(c[269], c[269] * ramp, 30)
        df["close"] = c
        df["open"] = np.r_[c[0], c[:-1]]
        df["high"] = df["close"] * 1.004
        df["low"] = df["close"] * 0.996
        _, tr = supertrend(df["high"], df["low"], df["close"],
                           settings.TREND_ST_PERIOD, settings.TREND_ST_MULT)
        flips = np.where((tr == 1) & (tr.shift(1) == -1))[0]
        if not len(flips):
            continue
        df = df.iloc[: int(flips[-1]) + 1].reset_index(drop=True)
        ema = df["close"].ewm(span=settings.TREND_EMA_PERIOD, adjust=False,
                              min_periods=settings.TREND_EMA_PERIOD).mean()
        if float(df["close"].iloc[-1]) > float(ema.iloc[-1]):
            return df
    raise AssertionError("no grid combo produced flip-above-EMA200")


@pytest.fixture()
def scanner(monkeypatch):
    s = TrendScanner()
    s._last_fired.clear()
    return s


def _serve(monkeypatch, df: pd.DataFrame):
    monkeypatch.setattr(trend_scanner_module, "data_fetcher",
                        type("F", (), {"get_candles": staticmethod(
                            lambda *a, **k: df.copy())})())


# ============================================================
# Scanner: Donchian breakout
# ============================================================

def test_donchian_fires_on_cross_with_volume(monkeypatch, scanner):
    df = _breakout_frame()
    _serve(monkeypatch, df)
    out = scanner._scan_one("TESTUSDT")
    srcs = [o["source"] for o in out]
    assert "trend_donchian" in srcs
    cand = next(o for o in out if o["source"] == "trend_donchian")
    price = float(df["close"].iloc[-1])
    a = cand["atr"]
    # research geometry: SL 2.0xATR, TP1 2.4xATR, TP2 4.8xATR
    assert cand["stop_loss"] == pytest.approx(price - 2.0 * a, rel=1e-6)
    assert cand["take_profit"] == pytest.approx(price + 2.4 * a, rel=1e-6)
    assert cand["take_profit_2"] == pytest.approx(price + 4.8 * a, rel=1e-6)
    assert cand["risk_reward_ratio"] >= settings.MIN_RR_RATIO
    assert cand["score"] >= settings.TREND_MIN_SCORE


def test_donchian_silent_without_volume_thrust(monkeypatch, scanner):
    df = _breakout_frame()
    df.iloc[-1, df.columns.get_loc("volume")] = 1000.0  # exactly average
    _serve(monkeypatch, df)
    out = scanner._scan_one("TESTUSDT")
    assert "trend_donchian" not in [o["source"] for o in out]


def test_donchian_never_fires_inside_channel(monkeypatch, scanner):
    df = _base_frame()  # pure sideways, no cross
    _serve(monkeypatch, df)
    out = scanner._scan_one("TESTUSDT")
    assert "trend_donchian" not in [o["source"] for o in out]


def test_symbol_cooldown_blocks_refire(monkeypatch, scanner):
    df = _breakout_frame()
    _serve(monkeypatch, df)
    out1 = scanner._scan_one("TESTUSDT")
    assert any(o["source"] == "trend_donchian" for o in out1)
    out2 = scanner._scan_one("TESTUSDT")
    assert "trend_donchian" not in [o["source"] for o in out2]


def test_disabled_channel_returns_empty(monkeypatch, scanner):
    monkeypatch.setattr(settings, "TREND_BOOST_ENABLED", False)
    assert scanner.scan(["TESTUSDT"]) == []


# ============================================================
# Scanner: SuperTrend flip (production indicator)
# ============================================================

def test_st_flip_fires_above_ema200(monkeypatch, scanner):
    df = _flip_frame()
    # self-verify the fixture: the PRODUCTION supertrend flips up on the
    # last closed bar while price is above EMA200
    _, tr = supertrend(df["high"], df["low"], df["close"],
                       settings.TREND_ST_PERIOD, settings.TREND_ST_MULT)
    ema = df["close"].ewm(span=settings.TREND_EMA_PERIOD, adjust=False,
                          min_periods=settings.TREND_EMA_PERIOD).mean()
    assert bool((tr.iloc[-1] == 1) and (tr.iloc[-2] == -1)), \
        "fixture must end on a bullish SuperTrend flip"
    assert float(df["close"].iloc[-1]) > float(ema.iloc[-1]), \
        "fixture flip must be above EMA200"
    _serve(monkeypatch, df)
    out = scanner._scan_one("TESTUSDT")
    assert "trend_st_flip" in [o["source"] for o in out]


# ============================================================
# Market gates
# ============================================================

def test_dead_market_gate(monkeypatch, scanner):
    df = _breakout_frame()
    # crush realised vol: constant closes + tiny bands -> ATR% far below
    # the 0.30% floor (the gate runs BEFORE any signal logic)
    df["close"] = 100.0
    df["open"] = 100.0
    df["high"] = 100.05
    df["low"] = 99.95
    _serve(monkeypatch, df)
    out = scanner._scan_one("TESTUSDT")
    assert out == []


# ============================================================
# Rec builder contract
# ============================================================

def test_build_trend_rec_contract():
    cand = {
        "symbol": "TESTUSDT", "source": "trend_donchian", "score": 76,
        "direction": "bullish", "timeframe": "4h",
        "current_price": 100.0, "stop_loss": 98.0, "take_profit": 102.4,
        "take_profit_2": 104.8, "risk_reward_ratio": 2.4, "atr": 1.0,
        "atr_pct": 1.0, "volume_ratio": 2.0, "above_ema200": True,
        "signals": ["r1", "r2", "r3"], "analyzed_at": "now",
    }
    rec = build_trend_rec(cand)
    assert rec["boosted_from_trend"] is True
    assert rec["direction"] == "bullish"
    # confidence cap: genuine strategy signals still rank first
    assert rec["confidence"] == settings.TREND_CONF_CAP
    assert rec["admission_confidence"] == settings.TREND_CONF_CAP
    assert rec["harmony"] == pytest.approx(0.55)
    assert rec["risk_reward_ratio"] >= settings.MIN_RR_RATIO
    assert rec["expected_rise_pct"] >= settings.MIN_EXPECTED_RISE
    assert rec["signals"][0]["strategy"] == "trend_donchian"


def test_build_trend_rec_conf_cap():
    cand = {"symbol": "T", "source": "trend_st_flip", "score": 99,
            "current_price": 10.0, "stop_loss": 9.8,
            "take_profit": 10.24, "take_profit_2": 10.48,
            "risk_reward_ratio": 2.4, "atr": 0.1, "atr_pct": 1.0,
            "signals": ["only"], "volume_ratio": 1.5,
            "above_ema200": True, "timeframe": "4h"}
    rec = build_trend_rec(cand)
    assert rec["confidence"] <= settings.TREND_CONF_CAP


# ============================================================
# Risk manager channel caps
# ============================================================

@pytest.fixture()
def ledger_db(tmp_path):
    from src.db.database import Database
    return Database(db_url=None, db_path=tmp_path / "ledger.db")


@pytest.fixture()
def rm(tmp_path, monkeypatch, ledger_db):
    import src.risk.manager as manager_module
    pos_file = tmp_path / "open_positions.json"
    stats_file = tmp_path / "daily_stats.json"
    pending_file = tmp_path / "pending_entries.json"
    pos_file.write_text("[]")
    stats_file.write_text("{}")
    pending_file.write_text("[]")
    monkeypatch.setattr(manager_module, "POSITIONS_FILE", pos_file)
    monkeypatch.setattr(manager_module, "DAILY_STATS_FILE", stats_file)
    monkeypatch.setattr(manager_module, "PENDING_FILE", pending_file)
    monkeypatch.setattr(manager_module, "db", ledger_db)
    from src.risk.manager import RiskManager
    return RiskManager(capital=10000)


def _trend_pos(entry_min_ago: float) -> dict:
    from src.utils.helpers import now_utc
    from datetime import timedelta
    t = now_utc() - timedelta(minutes=entry_min_ago)
    return {"symbol": "XUSDT", "strategy": "trend_donchian",
            "boosted_from_trend": True,
            "entry_time": t.isoformat()}


def test_trend_channel_concurrent_cap(rm):
    rm.open_positions = [_trend_pos(999) for _ in
                         range(settings.TREND_MAX_OPEN_CONCURRENT)]
    ok, why = rm._trend_channel_ok("NEWUSDT")
    assert not ok and "concurrent cap" in why


def test_trend_channel_entry_spacing(rm):
    rm.open_positions = [_trend_pos(settings.TREND_ENTRY_SPACING_MIN / 2)]
    ok, why = rm._trend_channel_ok("NEWUSDT")
    assert not ok and "spacing" in why


def test_trend_channel_ok_when_clear(rm):
    rm.open_positions = [_trend_pos(settings.TREND_ENTRY_SPACING_MIN * 5)]
    ok, why = rm._trend_channel_ok("NEWUSDT")
    assert ok and why == ""


def test_trend_rec_passes_validate(rm):
    rec = build_trend_rec({
        "symbol": "TESTUSDT", "source": "trend_donchian", "score": 70,
        "current_price": 100.0, "stop_loss": 98.0, "take_profit": 102.4,
        "take_profit_2": 104.8, "risk_reward_ratio": 2.4, "atr": 1.0,
        "atr_pct": 1.0, "signals": ["r1", "r2"], "volume_ratio": 2.0,
        "above_ema200": True, "timeframe": "4h"})
    rec["volatility_extreme"] = False
    rec["dead_market"] = False
    rec["session_blocked"] = False
    rec["decision"] = {"vetoed": False, "veto_reason": "",
                       "adjustments": [], "a_plus": False,
                       "base_confidence": rec["confidence"]}
    rec["harmony"] = 0.55
    valid, reasons = rm.validate_recommendation(rec)
    assert valid, reasons


# ============================================================
# Composite stack: retirements + starvation fix
# ============================================================

def test_default_composite_stack_excludes_retired():
    from src.analysis.scorer import SignalScorer
    names = [type(s).__name__ for s in SignalScorer().strategies]
    assert "LiquiditySweepReversalStrategy" not in names
    assert "TrendPullbackStrategy" not in names
    assert "VolatilityBreakoutStrategy" in names


def test_retired_composites_can_be_reenabled():
    from src.analysis.scorer import SignalScorer
    old_t = settings.STRATEGY_TREND_PULLBACK_ENABLED
    old_l = settings.STRATEGY_LIQUIDITY_SWEEP_ENABLED
    try:
        settings.STRATEGY_TREND_PULLBACK_ENABLED = True
        settings.STRATEGY_LIQUIDITY_SWEEP_ENABLED = True
        names = [type(s).__name__ for s in SignalScorer().strategies]
        assert "TrendPullbackStrategy" in names
        assert "LiquiditySweepReversalStrategy" in names
    finally:
        settings.STRATEGY_TREND_PULLBACK_ENABLED = old_t
        settings.STRATEGY_LIQUIDITY_SWEEP_ENABLED = old_l


def test_confluence_ref_weight_starvation_fix():
    # solo vote of the heaviest remaining composite (1.8) maps to
    # confluence 0.90, so a full signal at strength ~57 clears
    # MIN_CONFIDENCE=68 (was strength >= 86 under REF=5.8)
    assert settings.STRATEGY_CONFLUENCE_REF_WEIGHT == pytest.approx(2.0)
    conf = 100 * (0.65 * 0.57 + 0.35 * min(1.8 / 2.0, 1.0))
    assert conf >= settings.MIN_CONFIDENCE
    # and the old scale would have failed the same signal
    old_conf = 100 * (0.65 * 0.57 + 0.35 * min(1.8 / 5.8, 1.0))
    assert old_conf < settings.MIN_CONFIDENCE
