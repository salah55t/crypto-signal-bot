"""
v5.20 "Bounce Realist" - the giveback fix (production forensics #4).

Production evidence (6 closed trades 2026-09-29/30, ALL bottom_scanner_boost,
ALL "Stop Loss Hit", profit factor 0.24, avg capture efficiency -50%):
  SPCX  +0.49% (the ONLY TP1 fill - its 1.2x-ATR TP1 was just 1.12% away)
  INTCB -2.59% over 11.6h with MFE 0.22%  (dead capital, 72h time stop)
  MORPHO +0.80% from a +2.04% peak        (ladder gave back 60%)
  ZAMA  -5.71% (4.6% ATR -> 5.5% stop, MFE 0.11%) = 79% of the net loss
  CAKE  +0.80% from a +2.86% peak (missed TP1 by 0.05%!)
  ZEC   -0.20% from a +1.95% peak (old +1% -> flat BE lock gave back all)

Root causes (verified in code):
  1. No MAX-ATR gate on the bottom channel -> 4.6%-ATR knives with 5.5%
     stops. 46% of the day's candidates had ATR% > 3.
  2. TP1 = 1.2x ATR(4h) sat 2.9-5.5% away while the bounce MFEs died at
     0.11-2.86% (median 1.6%) -> fill rate 1/6, on the lowest-ATR coin.
  3. Coarse ladder (+1% -> flat BE, +2% -> +1%) returned 50-100% of every
     bounce; chandelier 2.5x ATR(4h) = 6-11% below peak -> never armed.
  4. Entry clustering: 5 bottom longs in 6h on a falling tape (per-cycle
     cap cannot see cross-cycle stacking; no BTC-tide check for bottoms).
  5. build_bottom_rec pinned EVERY confidence at exactly 72.0 -> ranking,
     regime corridor and continuation logic all blind.
  6. _position_from_row dropped boosted_from_bottom -> every Render
     redeploy silently stripped the v5.19 immunity from surviving trades.

Fixes covered:
  - bottom_scanner: BOTTOM_MAX_ATR_PCT ceiling + TP1 cap
    min(1.2x ATR, BOTTOM_TP1_CAP_PCT).
  - manager ladder: settings-driven finer rungs (+1% -> +0.30%, +2% ->
    +1.10%), mirrored for shorts.
  - manager._check_time_stop: bottom stagnation exit (6h, no bounce).
  - manager._bottom_channel_ok: concurrent cap + entry spacing + BTC tide.
  - analyzer.build_bottom_rec: differentiated confidence (late-chase +
    noise penalties before the cap).
  - manager._position_from_row: restores boosted_from_bottom from strategy.
"""
import importlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from src.analysis.analyzer import build_bottom_rec
from src.analysis.bottom_scanner import score_bottom_candidate
from src.risk.manager import RiskManager
import src.risk.manager as manager_module

cfgmod = importlib.import_module("config.settings")


def _mod(name: str):
    return importlib.import_module(name)


def _df(closes, opens=None, volumes=None, freq="4h"):
    n = len(closes)
    closes = np.asarray(closes, dtype=float)
    if opens is None:
        opens = np.concatenate([[closes[0]], closes[:-1]])
    opens = np.asarray(opens, dtype=float)
    highs = np.maximum(opens, closes) * 1.0005
    lows = np.minimum(opens, closes) * 0.9995
    vol = np.full(n, 1000.0) if volumes is None else np.asarray(volumes, float)
    idx = pd.date_range("2026-09-20", periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({"open": opens, "high": highs, "low": lows,
                         "close": closes, "volume": vol}, index=idx)


def _candidate(score=65.0, n_signals=3, symbol="BOTTOMUSDT", atr_pct=1.2,
               dist_from_low=2.0):
    """A bottom-scanner candidate as score_bottom_candidate returns it."""
    return {
        "symbol": symbol,
        "score": float(score),
        "current_price": 1.0,
        "recent_low": 0.98,
        "recent_high": 1.30,
        "distance_from_low_pct": dist_from_low,
        "position_in_range_pct": 6.7,
        "rsi": 28.0,
        "atr_pct": atr_pct,
        "stop_loss": 1.0 - 1.2 * (atr_pct / 100.0),
        "take_profit": 1.0 + 2.5 * (atr_pct / 100.0),
        "risk_reward_ratio": 2.083,
        "signals": [f"layer{i}" for i in range(n_signals)],
        "bb_percent_b": 0.03,
        "last_candle_bullish": True,
        "patterns_detected": ["hammer"],
        "analyzed_at": "2026-09-25T00:00:00+00:00",
    }


from tests.test_v5_veteran import make_manager, make_position


# ======================================================================
# A) MAX-ATR gate: the ZAMA failure (5.5% stop, 79% of the net loss)
# ======================================================================
def _dump_df(atr_pct_target):
    """Steepness-tuned dump: per-bar move of atr_pct_target%% plus tiny
    bullish closes (the last candle must close up for the scanner). Keeps
    the realized ATR within ~0.1pp of the target so gate tests are exact."""
    move = atr_pct_target / 100.0
    closes = []
    p = 1.50
    for i in range(58):
        closes.append(p)
        p *= (1.0 - move)
    closes += [p * (1 + move * 0.1), p * (1 + move * 0.2),
               p * (1 + move * 0.3)]
    return _df(closes)


def test_max_atr_gate_skips_zama_like_knife():
    """ATR ~4.6% (ZAMAUSDT) must be rejected: bounces die at 1.6% median
    while the stop would sit 5.5% away."""
    r = score_bottom_candidate("ZAMALIKE", _dump_df(4.6))
    assert r.get("skip"), "high-ATR knife must be skipped"
    assert "ceiling" in (r.get("reason") or "")


def test_normal_atr_coin_still_passes():
    """A ~1.5%-ATR dump (the healthy case) still passes the new ceiling."""
    r = score_bottom_candidate("DUMPUSDT", _dump_df(1.5))
    assert not r.get("skip"), f"must pass gates, got: {r.get('reason')}"
    assert r["atr_pct"] <= settings.BOTTOM_MAX_ATR_PCT


def test_settings_v520_exist():
    assert settings.BOTTOM_MAX_ATR_PCT == pytest.approx(3.0)
    assert settings.BOTTOM_TP1_CAP_PCT == pytest.approx(1.5)
    assert settings.BOTTOM_MAX_OPEN_CONCURRENT == 2
    assert settings.BOTTOM_ENTRY_SPACING_MIN == pytest.approx(45.0)
    assert settings.BOTTOM_STAGNATION_HOURS == pytest.approx(6.0)
    assert settings.LADDER_LOCK1_LEVEL_PCT > 0.0   # no more flat BE rung
    assert (settings.LADDER_LOCK1_LEVEL_PCT < settings.LADDER_LOCK2_LEVEL_PCT
            < settings.LADDER_LOCK3_LEVEL_PCT)


# ======================================================================
# B) TP1 cap: the bank leg becomes reachable (was 1/6 fill)
# ======================================================================
def test_tp1_capped_on_high_atr_coins():
    """Realized ATR ~2.7% -> 1.2x ATR = 3.3% > 1.5% cap -> TP1 pinned at
    1.5% of price (the CAKE case: peak +2.86% would then BANK)."""
    r = score_bottom_candidate("CAKELIKE", _dump_df(2.2))
    assert not r.get("skip"), f"must pass gates, got: {r.get('reason')}"
    price = r["current_price"]
    expected = price * (1 + settings.BOTTOM_TP1_CAP_PCT / 100.0)
    assert r["take_profit"] == pytest.approx(expected, rel=1e-6)
    # TP2 stays the ATR runner
    atr = price * r["atr_pct"] / 100.0
    assert r["take_profit_2"] == pytest.approx(price + 2.5 * atr, rel=1e-6)


def test_tp1_uncapped_on_low_atr_coins():
    """ATR ~1.0% -> 1.2x ATR = 1.2% < 1.5% cap -> untouched (the SPCX
    case that DID fill)."""
    r = score_bottom_candidate("SPCXLIKE", _dump_df(0.9))
    assert not r.get("skip"), f"must pass gates, got: {r.get('reason')}"
    price = r["current_price"]
    atr = price * r["atr_pct"] / 100.0
    assert r["take_profit"] == pytest.approx(price + 1.2 * atr, rel=1e-6)


def test_rr_gate_still_on_tp2():
    """RR must keep being computed on the runner, not on the capped TP1."""
    r = score_bottom_candidate("CAKELIKE", _dump_df(2.2))
    assert not r.get("skip")
    assert r["risk_reward_ratio"] == pytest.approx(2.5 / 1.2, rel=1e-3)


# ======================================================================
# C) Finer ladder: the ZEC replay (+1.95% peak must lock, not round-trip)
# ======================================================================
def test_ladder_lock1_no_longer_flat_breakeven(tmp_path, monkeypatch):
    """ZEC replay: +1.95% peak -> SL must sit at entry+0.30%, NOT entry."""
    monkeypatch.setattr(settings, "CHANDELIER_ENABLED", False)
    rm = make_manager(tmp_path, monkeypatch)
    pos = make_position(entry=1405.32, sl=1337.23, tp=1473.41)
    pos["atr"] = 1405.32 * 0.04  # big ATR -> chandelier stays out of the way
    rm.open_positions = [pos]
    updates = rm.apply_trailing_logic({"TESTUSDT": 1432.79}, {})  # +1.95%
    assert updates, "ladder must raise the SL"
    expected = 1405.32 * (1 + settings.LADDER_LOCK1_LEVEL_PCT / 100.0)
    assert pos["stop_loss"] == pytest.approx(expected)
    assert pos["stop_loss"] > 1405.32, "lock must be ABOVE entry (not BE)"


def test_ladder_lock2_tighter_than_old(tmp_path, monkeypatch):
    """MORPHO replay: +2.04% peak -> lock +1.10% (was +1.00%)."""
    monkeypatch.setattr(settings, "CHANDELIER_ENABLED", False)
    rm = make_manager(tmp_path, monkeypatch)
    pos = make_position(entry=2.499, sl=2.389, tp=2.609)
    pos["atr"] = 2.499 * 0.04
    rm.open_positions = [pos]
    updates = rm.apply_trailing_logic({"TESTUSDT": 2.55}, {})  # +2.04%
    assert updates
    expected = 2.499 * (1 + settings.LADDER_LOCK2_LEVEL_PCT / 100.0)
    assert pos["stop_loss"] == pytest.approx(expected)
    assert pos["stop_loss"] > 2.499 * 1.01, "must beat the old +1% rung"


def test_ladder_mirror_for_shorts(tmp_path, monkeypatch):
    """Bearish mirror: -1.5% profit locks -0.30% below entry."""
    monkeypatch.setattr(settings, "CHANDELIER_ENABLED", False)
    rm = make_manager(tmp_path, monkeypatch)
    pos = make_position(entry=100.0, sl=102.0, tp=96.0, direction="bearish")
    pos["atr"] = 4.0
    rm.open_positions = [pos]
    updates = rm.apply_trailing_logic({"TESTUSDT": 98.5}, {})  # +1.5%
    assert updates
    expected = 100.0 * (1 - settings.LADDER_LOCK1_LEVEL_PCT / 100.0)
    assert pos["stop_loss"] == pytest.approx(expected)
    assert pos["stop_loss"] < 100.0


def test_ladder_uses_settings_thresholds(tmp_path, monkeypatch):
    """Rung thresholds are env-tunable: +0.9% with LOCK1_PCT=0.8 locks."""
    monkeypatch.setattr(settings, "CHANDELIER_ENABLED", False)
    monkeypatch.setattr(settings, "LADDER_LOCK1_PCT", 0.8)
    rm = make_manager(tmp_path, monkeypatch)
    pos = make_position(entry=100.0, sl=98.0, tp=104.0)
    pos["atr"] = 5.0
    rm.open_positions = [pos]
    updates = rm.apply_trailing_logic({"TESTUSDT": 100.9}, {})
    assert updates
    assert pos["stop_loss"] == pytest.approx(
        100.0 * (1 + settings.LADDER_LOCK1_LEVEL_PCT / 100.0))


# ======================================================================
# D) Bottom stagnation exit: the INTCB/ZAMA dead-capital replay
# ======================================================================
def test_stagnation_exit_kills_dead_bottom_bounce(tmp_path, monkeypatch):
    """7h-old bottom trade, pnl +0.1%, MFE 0.22% (INTCB profile) -> close."""
    monkeypatch.setattr(settings, "STRUCTURAL_EXITS_ENABLED", False)
    rm = make_manager(tmp_path, monkeypatch)
    pos = make_position(entry=118.47, sl=115.63, tp=121.31)
    pos["boosted_from_bottom"] = True
    pos["mfe_pct"] = 0.22
    pos["entry_time"] = (datetime.now(timezone.utc)
                         - timedelta(hours=7)).isoformat()
    rm.open_positions = [pos]
    results = rm.check_open_positions({"TESTUSDT": 118.59})  # +0.10%
    assert results, "stagnation exit must fire"
    assert "Bottom stagnation" in results[-1].get("reason", "")


def test_stagnation_exit_spares_live_bounce(tmp_path, monkeypatch):
    """Same age but MFE 1.2% -> the bounce appeared, keep managing."""
    monkeypatch.setattr(settings, "STRUCTURAL_EXITS_ENABLED", False)
    rm = make_manager(tmp_path, monkeypatch)
    pos = make_position(entry=118.47, sl=116.0, tp=121.31)
    pos["boosted_from_bottom"] = True
    pos["mfe_pct"] = 1.2
    pos["entry_time"] = (datetime.now(timezone.utc)
                         - timedelta(hours=7)).isoformat()
    rm.open_positions = [pos]
    results = rm.check_open_positions({"TESTUSDT": 118.59})
    assert results == []


def test_stagnation_exit_market_trades_keep_72h_horizon(tmp_path, monkeypatch):
    """Non-boosted position with the same stale profile -> NOT stagnation-
    killed (72h horizon untouched for market entries)."""
    monkeypatch.setattr(settings, "STRUCTURAL_EXITS_ENABLED", False)
    rm = make_manager(tmp_path, monkeypatch)
    pos = make_position(entry=118.47, sl=115.63, tp=121.31)
    pos["boosted_from_bottom"] = False
    pos["mfe_pct"] = 0.22
    pos["entry_time"] = (datetime.now(timezone.utc)
                         - timedelta(hours=7)).isoformat()
    rm.open_positions = [pos]
    results = rm.check_open_positions({"TESTUSDT": 118.59})
    assert results == []


# ======================================================================
# E) Clustering caps: the 5-entries-in-6h burst replay
# ======================================================================
def _bottom_rec(symbol="NEWUSDT"):
    rec = build_bottom_rec(_candidate(symbol=symbol))
    rec["take_profit_2"] = rec["take_profit"] * 1.05
    return rec


def test_concurrent_bottom_cap_blocks_third(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "STRUCTURAL_EXITS_ENABLED", False)
    rm = make_manager(tmp_path, monkeypatch)
    for s in ("BOT1USDT", "BOT2USDT"):
        p = make_position(symbol=s)
        p["boosted_from_bottom"] = True
        rm.open_positions.append(p)
    result = rm.open_paper_position(_bottom_rec("NEWUSDT"))
    assert result["status"] == "rejected"
    assert any("concurrent cap" in r for r in result["reasons"])


def test_entry_spacing_blocks_rapid_refire(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "STRUCTURAL_EXITS_ENABLED", False)
    rm = make_manager(tmp_path, monkeypatch)
    p = make_position(symbol="BOT1USDT")
    p["boosted_from_bottom"] = True
    p["entry_time"] = (datetime.now(timezone.utc)
                       - timedelta(minutes=20)).isoformat()
    rm.open_positions.append(p)
    result = rm.open_paper_position(_bottom_rec("NEWUSDT"))
    assert result["status"] == "rejected"
    assert any("spacing" in r for r in result["reasons"])


def test_spacing_allows_entries_after_gap(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "STRUCTURAL_EXITS_ENABLED", False)
    monkeypatch.setattr(cfgmod.settings, "MIN_RR_RATIO", 1.5)
    rm = make_manager(tmp_path, monkeypatch)
    p = make_position(symbol="BOT1USDT")
    p["boosted_from_bottom"] = True
    p["entry_time"] = (datetime.now(timezone.utc)
                       - timedelta(hours=2)).isoformat()
    rm.open_positions.append(p)
    result = rm.open_paper_position(_bottom_rec("NEWUSDT"))
    assert result["status"] == "opened", result.get("reasons")


def test_market_entries_bypass_bottom_caps(tmp_path, monkeypatch):
    """The caps are channel-scoped: a market rec must open even with two
    bottom positions already running."""
    monkeypatch.setattr(settings, "STRUCTURAL_EXITS_ENABLED", False)
    monkeypatch.setattr(cfgmod.settings, "MIN_RR_RATIO", 1.5)
    monkeypatch.setattr(cfgmod.settings, "MIN_CONFIDENCE", 60.0)
    monkeypatch.setattr(cfgmod.settings, "MIN_EXPECTED_RISE", 1.0)
    rm = make_manager(tmp_path, monkeypatch)
    for s in ("BOT1USDT", "BOT2USDT"):
        p = make_position(symbol=s)
        p["boosted_from_bottom"] = True
        rm.open_positions.append(p)
    rec = _bottom_rec("MKTUSDT")
    rec["boosted_from_bottom"] = False
    rec["harmony"] = 0.9
    result = rm.open_paper_position(rec)
    assert result["status"] == "opened", result.get("reasons")


def test_btc_tide_gate_blocks_bottom_longs(tmp_path, monkeypatch):
    """Bearish 1h BTC regime -> no new counter-tape bounce longs."""
    monkeypatch.setattr(settings, "STRUCTURAL_EXITS_ENABLED", False)
    rm = make_manager(tmp_path, monkeypatch)
    # AFTER make_manager (it disables the gate for suite isolation)
    monkeypatch.setattr(settings, "BOTTOM_BTC_TIDE_GATE", True)
    monkeypatch.setattr(rm, "_tide_snapshot", lambda: ("bearish", -55.0))
    result = rm.open_paper_position(_bottom_rec("NEWUSDT"))
    assert result["status"] == "rejected"
    assert any("tide gate" in r for r in result["reasons"])


def test_btc_tide_gate_neutral_regime_allows(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "STRUCTURAL_EXITS_ENABLED", False)
    monkeypatch.setattr(cfgmod.settings, "MIN_RR_RATIO", 1.5)
    rm = make_manager(tmp_path, monkeypatch)
    monkeypatch.setattr(settings, "BOTTOM_BTC_TIDE_GATE", True)
    monkeypatch.setattr(rm, "_tide_snapshot", lambda: ("neutral", -10.0))
    result = rm.open_paper_position(_bottom_rec("NEWUSDT"))
    assert result["status"] == "opened", result.get("reasons")


# ======================================================================
# F) Confidence differentiation: no more wall of 72.0
# ======================================================================
def test_confidence_penalized_for_late_chase_and_noise():
    """dist 4.9% + ATR 4.6% (ZAMA profile) must score BELOW a clean
    candidate; the cap alone cannot tell them apart."""
    clean = build_bottom_rec(_candidate(dist_from_low=1.0, atr_pct=1.2))
    zama = build_bottom_rec(_candidate(dist_from_low=4.9, atr_pct=4.6))
    assert zama["confidence"] < clean["confidence"]
    assert clean["confidence"] == pytest.approx(settings.BOTTOM_CONF_CAP)
    # and the penalty is real, not cosmetic
    raw = 40.0 + 65.0 / 2.0
    assert zama["confidence"] < raw - 5.0


def test_confidence_penalties_never_negative():
    rec = build_bottom_rec(_candidate(dist_from_low=99.0, atr_pct=99.0))
    assert rec["confidence"] >= 0.0


# ======================================================================
# G) Ledger restore keeps the bottom flag (redeploy survival)
# ======================================================================
def test_position_from_row_restores_boosted_flag():
    row = {
        "symbol": "ZECUSDT", "direction": "bullish", "entry_price": 1405.32,
        "stop_loss": 1337.23, "take_profit": 1473.41, "tp2": 1547.18,
        "size": 0.006, "notional_usd": 8.5, "entry_fee": 0.0085,
        "entry_time": "2026-09-29T21:01:30+00:00", "confidence": 72.0,
        "paper": 1, "trade_uid": "TRD-X", "initial_sl": 1337.23,
        "initial_tp": 1473.41, "tp1_taken": 0, "realized_pnl": 0.0,
        "partial_count": 0, "peak_price": 1405.32, "trough_price": 1405.32,
        "mfe_pct": 0.0, "mae_pct": 0.0,
        "strategy": "bottom_scanner_boost",
    }
    pos = RiskManager._position_from_row(row)
    assert pos["boosted_from_bottom"] is True


def test_position_from_row_market_strategy_not_boosted():
    row = {"symbol": "X", "strategy": "trend_pullback", "entry_price": 1.0,
           "stop_loss": 0.99, "take_profit": 1.05}
    pos = RiskManager._position_from_row(row)
    assert pos["boosted_from_bottom"] is False
