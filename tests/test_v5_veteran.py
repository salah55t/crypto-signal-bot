"""
v5 "Veteran Trader" - Unit Tests
Covers:
  - Partial TP at TP1 (fraction realized, SL -> BE+fees, TP2 promoted)
  - Partial closes don't touch win/loss streak counters
  - Full close after TP1 aggregates correctly
  - MFE/MAE excursion tracking
  - Time stop (stale trades) + absolute max holding
  - Structural exit decision (Ichimoku flip / opposite signal)
  - Chandelier trailing never loosens SL and respects the price cap
  - Pending limit entries: arm / fill at zone / cancel on zone break / expire
  - Harmony gate in validate_recommendation
  - Batched price endpoint weight map
"""
import sys
import json
from datetime import timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
import src.risk.manager as manager_module
from src.risk.manager import RiskManager
from src.utils.helpers import now_utc
from src.core.binance_client import _endpoint_weight
from src.core.rate_limiter import WeightedRateLimiter


def make_manager(tmp_path, monkeypatch, positions=None, daily=None):
    """RiskManager isolated from real data files."""
    pos_file = tmp_path / "open_positions.json"
    stats_file = tmp_path / "daily_stats.json"
    pending_file = tmp_path / "pending_entries.json"
    pos_file.write_text(json.dumps(positions if positions is not None else []))
    stats_file.write_text(json.dumps(daily or {}))
    pending_file.write_text(json.dumps([]))
    monkeypatch.setattr(manager_module, "POSITIONS_FILE", pos_file)
    monkeypatch.setattr(manager_module, "DAILY_STATS_FILE", stats_file)
    monkeypatch.setattr(manager_module, "PENDING_FILE", pending_file)

    class _NoDB:
        def __getattr__(self, name):
            return lambda *a, **k: None
    monkeypatch.setattr(manager_module, "db", _NoDB())
    return RiskManager(capital=10000)


def make_position(symbol="TESTUSDT", direction="bullish", notional=10.0,
                  entry=100.0, sl=98.0, tp=104.0, tp2=None, age_hours=0.0):
    return {
        "symbol": symbol,
        "direction": direction,
        "entry_price": entry,
        "stop_loss": sl,
        "take_profit": tp,
        "take_profit_2": tp2 if tp2 is not None else tp * 1.05,
        "size": notional / entry,
        "notional_usd": notional,
        "initial_notional_usd": notional,
        "entry_fee": notional * settings.TRADING_FEE_PCT / 100,
        "entry_time": (now_utc() - timedelta(hours=age_hours)).isoformat(),
        "harmony": 0.8,
        "atr": 1.0,
        "status": "open",
    }


# ============================================
# Partial TP
# ============================================

def test_tp1_partial_banks_half_and_moves_sl_to_breakeven(tmp_path, monkeypatch):
    rm = make_manager(tmp_path, monkeypatch)
    rm.open_positions = [make_position(tp=104.0, tp2=110.0)]
    prices = {"TESTUSDT": 105.0}
    results = rm.check_open_positions(prices)
    partials = [r for r in results if r.get("status") == "partial"]
    assert len(partials) == 1, "expected a TP1 partial close"
    pos = rm.open_positions[0]
    assert pos["tp1_taken"] is True
    # half the notional remains
    assert abs(pos["notional_usd"] - 5.0) < 1e-6
    # SL moved to break-even + fee buffer
    assert pos["stop_loss"] == pytest.approx(100.0 * (1 + settings.TP1_FEE_BUFFER_PCT / 100))
    # TP promoted to TP2
    assert pos["take_profit"] == pytest.approx(110.0)
    # partial PnL realized and positive
    assert partials[0]["pnl"] > 0


def test_partial_close_does_not_touch_streak(tmp_path, monkeypatch):
    rm = make_manager(tmp_path, monkeypatch)
    rm.open_positions = [make_position(tp=104.0, tp2=110.0)]
    rm.check_open_positions({"TESTUSDT": 105.0})
    today = rm.daily_stats[rm._today_key()]
    assert today.get("wins", 0) == 0 and today.get("losses", 0) == 0
    assert today.get("partials", 0) == 1


def test_full_close_after_tp1_hits_tp2(tmp_path, monkeypatch):
    rm = make_manager(tmp_path, monkeypatch)
    rm.open_positions = [make_position(tp=104.0, tp2=110.0)]
    rm.check_open_positions({"TESTUSDT": 105.0})   # partial
    results = rm.check_open_positions({"TESTUSDT": 111.0})  # TP2 hit
    fulls = [r for r in results if r.get("status") == "closed"]
    assert len(fulls) == 1
    assert fulls[0]["pnl"] > 0
    assert len(rm.open_positions) == 0
    assert rm.daily_stats[rm._today_key()]["wins"] == 1


def test_hard_sl_still_closes_fully(tmp_path, monkeypatch):
    rm = make_manager(tmp_path, monkeypatch)
    rm.open_positions = [make_position(sl=98.0)]
    results = rm.check_open_positions({"TESTUSDT": 97.0})
    assert results[0]["status"] == "closed"
    assert "Stop Loss" in results[0]["reason"]
    assert rm.daily_stats[rm._today_key()]["losses"] == 1


# ============================================
# MFE/MAE + time stop
# ============================================

def test_excursion_tracking_mfe_mae(tmp_path, monkeypatch):
    rm = make_manager(tmp_path, monkeypatch)
    pos = make_position()
    rm.track_excursions(pos, 106.0)
    rm.track_excursions(pos, 95.0)
    assert pos["peak_price"] == pytest.approx(106.0)
    assert pos["trough_price"] == pytest.approx(95.0)
    assert pos["mfe_pct"] == pytest.approx(6.0)
    assert pos["mae_pct"] == pytest.approx(5.0)


def test_time_stop_closes_stale_trade(tmp_path, monkeypatch):
    rm = make_manager(tmp_path, monkeypatch)
    rm.open_positions = [make_position(age_hours=settings.MAX_TRADE_HOURS + 1,
                                       tp=200.0)]
    results = rm.check_open_positions({"TESTUSDT": 100.4})
    assert len(results) == 1
    assert "Time stop" in results[0]["reason"]


def test_absolute_max_holding_closes_even_in_profit(tmp_path, monkeypatch):
    rm = make_manager(tmp_path, monkeypatch)
    rm.open_positions = [make_position(age_hours=settings.ABSOLUTE_MAX_TRADE_HOURS + 1,
                                       tp=200.0)]
    results = rm.check_open_positions({"TESTUSDT": 103.0})
    assert len(results) == 1
    assert "Max holding time" in results[0]["reason"]


def test_young_trade_never_time_stopped(tmp_path, monkeypatch):
    rm = make_manager(tmp_path, monkeypatch)
    rm.open_positions = [make_position(age_hours=1.0, tp=200.0)]
    results = rm.check_open_positions({"TESTUSDT": 100.5})
    assert results == []


# ============================================
# Structural exits
# ============================================

def test_structural_exit_on_regime_flip(tmp_path, monkeypatch):
    rm = make_manager(tmp_path, monkeypatch)
    pos = make_position()
    sig = {"direction": "bullish", "confidence": 70,
           "ichimoku": {"regime": "bearish", "score": -55}}
    action, reason = rm.evaluate_structural_exit(pos, sig, 101.0)
    assert action == "exit"
    assert "regime flipped bearish" in reason


def test_structural_exit_on_opposite_strong_signal(tmp_path, monkeypatch):
    rm = make_manager(tmp_path, monkeypatch)
    pos = make_position()
    sig = {"direction": "bearish", "confidence": 80, "ichimoku": {}}
    action, reason = rm.evaluate_structural_exit(pos, sig, 101.0)
    assert action == "exit"
    assert "Opposite" in reason


def test_structural_tighten_to_kijun(tmp_path, monkeypatch):
    rm = make_manager(tmp_path, monkeypatch)
    pos = make_position()
    sig = {"direction": "bullish", "confidence": 70,
           "ichimoku": {"regime": "neutral", "price_vs_kijun": "below",
                        "kijun": 99.5}}
    action, reason = rm.evaluate_structural_exit(pos, sig, 101.0)
    assert action == "tighten"
    assert "Kijun" in reason


def test_structural_none_when_aligned(tmp_path, monkeypatch):
    rm = make_manager(tmp_path, monkeypatch)
    pos = make_position()
    sig = {"direction": "bullish", "confidence": 70,
           "ichimoku": {"regime": "bullish", "price_vs_kijun": "above",
                        "kijun": 95.0, "tk_state": "bullish"}}
    action, reason = rm.evaluate_structural_exit(pos, sig, 101.0)
    assert action == "none"


# ============================================
# Chandelier trailing
# ============================================

def test_chandelier_trails_peak_with_atr(tmp_path, monkeypatch):
    rm = make_manager(tmp_path, monkeypatch)
    pos = make_position(entry=100.0, sl=98.0)
    pos["atr"] = 1.0
    rm.open_positions = [pos]
    # price rallied to 106; peak = 106 -> chandelier = 106 - 2.5*1 = 103.5
    updates = rm.apply_trailing_logic({"TESTUSDT": 105.5}, {})
    assert updates, "expected chandelier update"
    assert pos["stop_loss"] == pytest.approx(105.5 * 0.999, rel=1e-6) or \
           pos["stop_loss"] >= 103.0


def test_chandelier_disabled_no_trailing_below_breakeven(tmp_path, monkeypatch):
    rm = make_manager(tmp_path, monkeypatch)
    pos = make_position(entry=100.0, sl=98.0)
    pos["atr"] = 5.0  # huge ATR -> chandelier below SL -> no update
    rm.open_positions = [pos]
    updates = rm.apply_trailing_logic({"TESTUSDT": 101.5}, {})
    # profit 1.5% -> ladder wants BE (100); chandelier 101.5-12.5 < 98
    assert pos["stop_loss"] == pytest.approx(100.0)
    # ladder update happened (BE) but nothing above it
    assert all(u["new_sl"] <= 100.0 for u in updates if u["new_sl"])


def test_chandelier_never_above_current_price(tmp_path, monkeypatch):
    rm = make_manager(tmp_path, monkeypatch)
    pos = make_position(entry=100.0, sl=98.0)
    pos["atr"] = 0.1  # tiny ATR -> chandelier very close to peak
    rm.open_positions = [pos]
    price = 106.0
    pos["peak_price"] = 106.0
    rm.apply_trailing_logic({"TESTUSDT": price}, {})
    assert pos["stop_loss"] < price  # never at/above price


# ============================================
# Pending limit entries
# ============================================

def make_rec(symbol="TESTUSDT", price=110.0, zone=(100.0, 102.0), atr=2.0,
             harmony=0.8):
    return {
        "symbol": symbol, "direction": "bullish",
        "current_price": price, "entry_price": price,
        "stop_loss": 97.0, "take_profit": 112.0, "take_profit_2": 120.0,
        "entry_type": "limit", "entry_zone": {"low": zone[0], "high": zone[1]},
        "atr": atr, "harmony": harmony,
        "confidence": 75, "expected_rise_pct": 3.0, "risk_reward_ratio": 2.0,
        "decision": {},
    }


def test_harmony_gate_rejects_low_harmony(tmp_path, monkeypatch):
    rm = make_manager(tmp_path, monkeypatch)
    rec = make_rec(harmony=0.1)
    valid, reasons = rm.validate_recommendation(rec)
    assert not valid
    assert any("Harmony too low" in r for r in reasons)


def test_pending_entry_armed_and_fills_in_zone(tmp_path, monkeypatch):
    rm = make_manager(tmp_path, monkeypatch)
    rec = make_rec(price=110.0)  # far above zone (100-102)
    rm.add_pending_entry(rec, "test")
    assert len(rm.pending_entries) == 1
    # price returns into the zone -> fill
    filled = rm.check_pending_fills({"TESTUSDT": 101.5})
    assert len(filled) == 1
    assert filled[0]["fill_price"] == pytest.approx(101.5)
    assert len(rm.pending_entries) == 0
    assert len(rm.open_positions) == 1
    # filled at the ZONE price, not the chased price
    assert rm.open_positions[0]["entry_price"] == pytest.approx(101.5)


def test_pending_entry_cancelled_when_zone_breaks(tmp_path, monkeypatch):
    rm = make_manager(tmp_path, monkeypatch)
    rec = make_rec(price=110.0, atr=2.0)  # invalid level = 100 - 0.5*2 = 99
    rm.add_pending_entry(rec, "test")
    filled = rm.check_pending_fills({"TESTUSDT": 98.0})
    assert filled == []
    assert len(rm.pending_entries) == 0  # cancelled


def test_pending_entry_expires(tmp_path, monkeypatch):
    rm = make_manager(tmp_path, monkeypatch)
    rec = make_rec(price=110.0)
    pending = rm.add_pending_entry(rec, "test")
    pending["expires_at"] = (now_utc() - timedelta(hours=1)).isoformat()
    rm.check_pending_fills({"TESTUSDT": 110.0})
    assert len(rm.pending_entries) == 0
    assert len(rm.open_positions) == 0


# ============================================
# Rate limiter / weight map
# ============================================

def test_endpoint_weights():
    assert _endpoint_weight("/api/v3/klines", {}) == 2
    assert _endpoint_weight("/api/v3/depth", {"limit": 20}) == 5
    assert _endpoint_weight("/api/v3/ticker/24hr", {}) == 80          # full market
    assert _endpoint_weight("/api/v3/ticker/24hr", {"symbol": "BTCUSDT"}) == 2
    assert _endpoint_weight("/api/v3/ticker/price",
                            {"symbols": json.dumps(["BTCUSDT", "ETHUSDT"])}) == 2
    assert _endpoint_weight("/api/v3/exchangeInfo", {}) == 20


def test_rate_limiter_blocks_over_budget():
    # reserve=0 keeps the classic semantics (the reserve lane has its own
    # dedicated tests in test_v510_rate_priority.py)
    rl = WeightedRateLimiter(budget_per_min=10.0, priority_reserve=0.0)
    assert rl.acquire(6.0, timeout=0.1) is True
    assert rl.acquire(6.0, timeout=0.1) is False  # over budget -> refuse fast
    assert rl.used_weight() == pytest.approx(6.0)


def test_rate_limiter_cooldown_blocks_all():
    rl = WeightedRateLimiter(budget_per_min=6000.0)
    rl.trigger_cooldown(5.0)
    assert rl.in_cooldown()
    assert rl.acquire(1.0, timeout=0.1) is False


def test_tickers_batch_symbols_param_has_no_spaces():
    """Regression: json.dumps default separator ', ' -> %5B%22A%22,+%22B%22%5D
    is rejected by Binance (code -1100, illegal characters)."""
    from src.core.binance_client import binance_client
    captured = {}

    def fake_get(path, params, priority=False):
        captured["path"] = path
        captured["params"] = params
        return [{"symbol": "ACEUSDT", "price": "0.1793"}]

    original = binance_client._get
    binance_client._get = fake_get
    try:
        out = binance_client.get_tickers_batch(["ACEUSDT", "LTCUSDT", "RAYUSDT"])
    finally:
        binance_client._get = original
    assert out["ACEUSDT"]["price"] == "0.1793"
    sym = captured["params"]["symbols"]
    assert " " not in sym and "+" not in sym
    assert sym == '["ACEUSDT","LTCUSDT","RAYUSDT"]'
