"""
v5.19 "Bottom Survivor" - the last-churn fix (dashboard forensics #3).

Production evidence (2026-09-29, 16 closed trades):
  ALL 16 trades were bottom_scanner_boost (confidence exactly 72.0) and ALL
  closed with "Structural exit: Ichimoku regime flipped bearish" 30-40 min
  after entry - i.e. the instant the v5.18 grace expired. Net PnL +$0.11
  while fees ate $0.28 (fees = 40% of net); winners were killed mid-profit
  (INJUSDT +1.63%, UNIUSDT +1.02% at kill time).

Root cause chain (verified in code):
  1. build_bottom_rec() carries NO ichimoku field ->
     open_position records entry_ichimoku_regime=None for every bottom trade.
  2. evaluate_structural_exit: `None != "bearish"` is always True ->
     the v5.18 "genuine flip vs snapshot" guard was dead code for the only
     strategy that actually trades; every bearish reading became a "flip".
  3. A bottom lives BELOW the 4h cloud by definition (that is what a bottom
     is), so the bearish regime label is its natural state -> 100% kill at
     grace expiry.
  4. Geometry churn: single TP at 2.5x 4h-ATR (day-scale) while the average
     bounce MFE was 0.60% - trades covered 0-19% of the way to TP.
  5. Pending funnel: pendings armed 4.9-5.0 ATR below price with a 4h TTL
     (AAVE/NVDABUSDT) - mathematically unfillable, so every market-entry
     strategy looked "weak" while the ledger filled with doomed bottoms.

Fixes covered:
  - manager.evaluate_structural_exit: boosted_from_bottom positions are
    IMMUNE to regime-label exits (bullish + bearish mirrors); they exit on
    hard SL / opposite signal >= 55 / time stop / TP ladder only. Market
    entries keep the v5.18 flip semantics unchanged.
  - bottom_scanner: TP1/TP2 ladder (1.2x ATR bank + 2.5x ATR runner), RR
    still computed on TP2 so the MIN_RR_RATIO gate keeps its meaning.
  - analyzer.build_bottom_rec: maps take_profit_2 through to the rec.
  - manager.add_pending_entry: TTL scales with zone distance (1x..4x).
  - cycle.open_new_positions: PENDING_REACH_MAX_ATR drops unreachable zones.
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


def make_manager(tmp_path, monkeypatch):
    """RiskManager isolated from real data files (pattern of test_v5)."""
    from tests.test_v5_veteran import make_manager as _base
    return _base(tmp_path, monkeypatch)


# ======================================================================
# helpers
# ======================================================================
_BEAR_ICHO = {"regime": "bearish", "kijun": 104.0, "tenkan": 102.0,
              "price_vs_kijun": "below", "tk_cross_recent": "bearish",
              "tk_state": "bearish"}
_BULL_ICHO = {"regime": "bullish", "kijun": 96.0, "tenkan": 98.0,
              "price_vs_kijun": "above", "tk_cross_recent": "bullish",
              "tk_state": "bullish"}


def _bottom_pos(direction="bullish", age_min=120):
    """A bottom-fishing position exactly as production opens it:
    boosted_from_bottom=True and (because build_bottom_rec carries no
    ichimoku field) entry_ichimoku_regime=None."""
    return {
        "symbol": "BOTTOMUSDT",
        "direction": direction,
        "entry_time": (datetime.now(timezone.utc)
                       - timedelta(minutes=age_min)).isoformat(),
        "entry_ichimoku_regime": None,
        "boosted_from_bottom": True,
        "stop_loss": 98.8 if direction == "bullish" else 101.2,
    }


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


# ======================================================================
# A) THE fix: bottom trades survive the regime label (16/16 kill path)
# ======================================================================
def test_bottom_trade_survives_bearish_regime_after_grace(tmp_path,
                                                          monkeypatch):
    """THE production kill: bottom pos, no snapshot, bearish regime,
    grace long over. v5.18 closed all 16 trades exactly here."""
    rm = make_manager(tmp_path, monkeypatch)
    action, reason = rm.evaluate_structural_exit(
        _bottom_pos(),
        {"direction": "neutral", "confidence": 0, "ichimoku": _BEAR_ICHO},
        100.0)
    assert action == "none", (action, reason)


def test_bottom_trade_survives_repeatedly(tmp_path, monkeypatch):
    """Every future cycle must reach the same verdict - no flip roulette."""
    rm = make_manager(tmp_path, monkeypatch)
    sig = {"direction": "neutral", "confidence": 0,
           "ichimoku": _BEAR_ICHO}
    for _ in range(5):
        assert rm.evaluate_structural_exit(_bottom_pos(), sig, 100.0)[0] == "none"


def test_market_trade_without_snapshot_still_exits(tmp_path, monkeypatch):
    """Regression guard: NON-bottom positions keep the v5.18 semantics -
    a bearish regime with no entry snapshot still exits after grace."""
    rm = make_manager(tmp_path, monkeypatch)
    pos = _bottom_pos()
    pos.pop("boosted_from_bottom")
    action, reason = rm.evaluate_structural_exit(
        pos,
        {"direction": "neutral", "confidence": 0, "ichimoku": _BEAR_ICHO},
        100.0)
    assert action == "exit"
    assert reason == "Ichimoku regime flipped bearish"


def test_bottom_trade_opposite_signal_still_exits(tmp_path, monkeypatch):
    """The price thesis CAN die - on a full-analysis reversal (>= 55),
    never on the cloud label. That channel stays armed for bottoms."""
    rm = make_manager(tmp_path, monkeypatch)
    action, reason = rm.evaluate_structural_exit(
        _bottom_pos(),
        {"direction": "bearish", "confidence": 62, "ichimoku": _BULL_ICHO},
        100.0)
    assert action == "exit"
    assert "Opposite" in reason


def test_bottom_trade_hard_sl_still_armed(tmp_path, monkeypatch):
    """The bounce-fail price level (SL under the swing low) still closes
    the trade - immunity covers the LABEL, not the stop."""
    rm = make_manager(tmp_path, monkeypatch)
    pos = _bottom_pos()
    pos["entry_price"] = 100.0
    pos["take_profit"] = 101.2
    pos["take_profit_2"] = 102.0
    pos["size"] = 0.1
    pos["notional_usd"] = 10.0
    rm.open_positions = [pos]
    results = rm.check_open_positions({"BOTTOMUSDT": 98.0})
    assert len(results) == 1
    assert results[0]["reason"] == "Stop Loss Hit"


def test_short_bottom_trade_survives_bullish_regime(tmp_path, monkeypatch):
    """Mirror immunity for (future) short bottoms."""
    rm = make_manager(tmp_path, monkeypatch)
    action, _ = rm.evaluate_structural_exit(
        _bottom_pos(direction="bearish"),
        {"direction": "neutral", "confidence": 0, "ichimoku": _BULL_ICHO},
        100.0)
    assert action == "none"


def test_market_short_flip_still_exits(tmp_path, monkeypatch):
    """Mirror regression guard: non-bottom short keeps flip semantics."""
    rm = make_manager(tmp_path, monkeypatch)
    pos = _bottom_pos(direction="bearish")
    pos.pop("boosted_from_bottom")
    pos["entry_ichimoku_regime"] = "bearish"
    action, reason = rm.evaluate_structural_exit(
        pos,
        {"direction": "neutral", "confidence": 0, "ichimoku": _BULL_ICHO},
        100.0)
    assert action == "exit"
    assert reason == "Ichimoku regime flipped bullish"


# ======================================================================
# B) TP1/TP2 bounce ladder
# ======================================================================
def test_scanner_tp_ladder():
    """v5.20: TP1 = min(1.2x ATR, BOTTOM_TP1_CAP_PCT of price) - the bank
    leg must be REACHABLE (production: 1.2x-ATR TP1 sat 2.9-5.5% away while
    bounces died at MFE 0.11-2.86%, filled 1/6). TP2 keeps the runner
    (2.5x ATR); RR stays computed on TP2 (2.5/1.2 = 2.083 > MIN_RR_RATIO)."""
    closes = []
    p = 1.50
    for i in range(58):          # steady dump ~1.5% per bar
        closes.append(p)
        p *= 0.985
    closes += [p * 1.005, p * 1.010, p * 1.016]  # small bounce, bullish close
    r = score_bottom_candidate("DUMPUSDT", _df(closes))
    assert not r.get("skip"), f"must pass gates, got: {r.get('reason')}"
    price = r["current_price"]
    atr = price * r["atr_pct"] / 100.0
    # ~1.7% ATR puts 1.2x ATR (~2.0%) above the 1.5% cap -> capped
    tp1_expected = min(1.2 * atr, price * settings.BOTTOM_TP1_CAP_PCT / 100.0)
    assert r["take_profit"] == pytest.approx(price + tp1_expected)
    assert tp1_expected == pytest.approx(price * settings.BOTTOM_TP1_CAP_PCT / 100.0)
    assert r["take_profit_2"] == pytest.approx(price + 2.5 * atr)
    assert r["take_profit_2"] > r["take_profit"]
    assert r["risk_reward_ratio"] == pytest.approx(2.5 / 1.2, rel=1e-3)


def test_scanner_tp1_clears_fees_at_floor():
    """At the 0.60% ATR floor, TP1 = 1.2 x 0.60% = 0.72% > 0.2% round-trip
    fees - the bank leg is never fee-food."""
    assert (settings.BOTTOM_TP1_ATR_MULT * settings.BOTTOM_MIN_ATR_PCT) \
        > 2 * (settings.TRADING_FEE_PCT / 100.0) * 100.0


def test_build_bottom_rec_carries_tp2():
    c = {
        "symbol": "BOTTOMUSDT", "score": 65.0, "current_price": 1.0,
        "atr_pct": 1.2, "stop_loss": 0.9856,
        "take_profit": 1.0 + 1.2 * 0.012,
        "take_profit_2": 1.0 + 2.5 * 0.012,
        "risk_reward_ratio": 2.083,
        "signals": ["layer0", "layer1"], "last_candle_bullish": True,
    }
    rec = build_bottom_rec(c)
    assert rec["take_profit"] == pytest.approx(1.0 + 1.2 * 0.012)
    assert rec["take_profit_2"] == pytest.approx(1.0 + 2.5 * 0.012)
    assert rec["risk_reward_ratio"] == pytest.approx(2.083)
    assert rec["boosted_from_bottom"] is True


def test_build_bottom_rec_tp2_fallback_without_candidate_tp2():
    """Old candidates (no take_profit_2) degrade to the legacy single-TP
    behavior instead of crashing."""
    c = {
        "symbol": "BOTTOMUSDT", "score": 65.0, "current_price": 1.0,
        "atr_pct": 1.2, "stop_loss": 0.9856,
        "take_profit": 1.03, "risk_reward_ratio": 2.083,
        "signals": ["layer0"], "last_candle_bullish": True,
    }
    rec = build_bottom_rec(c)
    assert rec["take_profit_2"] == pytest.approx(rec["take_profit"])


def test_open_position_bottom_ladder_levels(tmp_path, monkeypatch):
    """End-to-end: the opened position banks at TP1 and promotes to TP2 -
    the veteran partial flow now has something near to bank."""
    rm = make_manager(tmp_path, monkeypatch)
    c = {
        "symbol": "BOTTOMUSDT", "score": 65.0, "current_price": 1.0,
        "atr_pct": 1.2, "stop_loss": 0.9856,
        "take_profit": 1.0 + 1.2 * 0.012,
        "take_profit_2": 1.0 + 2.5 * 0.012,
        "risk_reward_ratio": 2.083,
        "signals": ["layer0", "layer1", "layer2"],
        "last_candle_bullish": True,
    }
    rec = build_bottom_rec(c)
    result = rm.open_position(rec)
    assert result["status"] == "opened", result.get("reasons")
    pos = rm.open_positions[0]
    assert pos["take_profit"] == pytest.approx(1.0 + 1.2 * 0.012)
    assert pos["take_profit_2"] == pytest.approx(1.0 + 2.5 * 0.012)
    assert pos["initial_tp"] == pytest.approx(pos["take_profit"])
    assert pos["boosted_from_bottom"] is True
    # sanity: TP1 above entry, both targets above the stop
    assert pos["take_profit"] > pos["entry_price"] > pos["stop_loss"]
    assert pos["take_profit_2"] > pos["take_profit"]


# ======================================================================
# C) Pending reach: TTL scaling + unreachable-zone drop
# ======================================================================
def test_settings_v519_defaults():
    assert settings.BOTTOM_TP1_ATR_MULT == pytest.approx(1.2)
    assert settings.BOTTOM_TP2_ATR_MULT == pytest.approx(2.5)
    assert settings.BOTTOM_TP2_ATR_MULT > settings.BOTTOM_TP1_ATR_MULT
    assert settings.PENDING_REACH_MAX_ATR == pytest.approx(3.0)
    assert settings.PENDING_REACH_MAX_ATR > settings.PENDING_MOMENTUM_MAX_ATR


def _pending_rec():
    return {
        "symbol": "PENDUSDT", "direction": "bullish",
        "entry_type": "limit", "entry_zone": {"low": 100.0, "high": 102.0},
        "current_price": 110.0, "entry_price": 100.0,
        "stop_loss": 99.0, "take_profit": 110.0, "take_profit_2": 115.0,
        "atr": 2.0, "confidence": 90.0, "admission_confidence": 90.0,
        "expected_rise_pct": 3.0, "risk_reward_ratio": 2.0,
        "harmony": 0.9,
    }


def test_pending_ttl_scales_with_distance(tmp_path, monkeypatch):
    rm = make_manager(tmp_path, monkeypatch)
    p = rm.add_pending_entry(_pending_rec(), "test", dist_atr=2.0)
    created = datetime.fromisoformat(p["created_at"])
    expires = datetime.fromisoformat(p["expires_at"])
    assert (expires - created).total_seconds() == pytest.approx(
        settings.PENDING_TTL_HOURS * 2.0 * 3600.0, abs=60.0)


def test_pending_ttl_default_unchanged_without_distance(tmp_path,
                                                        monkeypatch):
    rm = make_manager(tmp_path, monkeypatch)
    p = rm.add_pending_entry(_pending_rec(), "test")
    created = datetime.fromisoformat(p["created_at"])
    expires = datetime.fromisoformat(p["expires_at"])
    assert (expires - created).total_seconds() == pytest.approx(
        settings.PENDING_TTL_HOURS * 3600.0, abs=60.0)


def test_pending_ttl_capped_at_4x(tmp_path, monkeypatch):
    rm = make_manager(tmp_path, monkeypatch)
    p = rm.add_pending_entry(_pending_rec(), "test", dist_atr=25.0)
    created = datetime.fromisoformat(p["created_at"])
    expires = datetime.fromisoformat(p["expires_at"])
    assert (expires - created).total_seconds() == pytest.approx(
        settings.PENDING_TTL_HOURS * 4.0 * 3600.0, abs=60.0)


# ======================================================================
# D) Live-data regression sketch: the production kill table
# ======================================================================
def test_production_kill_table_now_survives(tmp_path, monkeypatch):
    """Replay the exact 2026-09-29 production table: 13 trades died at
    30-40 min with regime bearish + snapshot None. Every one of them must
    now return "none" - the ledger can no longer monosex on this exit."""
    rm = make_manager(tmp_path, monkeypatch)
    symbols = ["BTCUSDT", "INTCBUSDT", "PENDLEUSDT", "INJUSDT", "UNIUSDT",
               "SPCXBUSDT", "CRCLBUSDT", "ZENUSDT", "XAUTUSDT", "TRXUSDT"]
    sig = {"direction": "neutral", "confidence": 0,
           "ichimoku": _BEAR_ICHO}
    for sym in symbols:
        pos = _bottom_pos()
        pos["symbol"] = sym
        action, _ = rm.evaluate_structural_exit(pos, sig, 100.0)
        assert action == "none", f"{sym} would still die on the label"
