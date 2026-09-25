"""
v5.11 Persistent Trade Ledger - Unit Tests

User incident: entry prices lived ONLY in data/open_positions.json (an
ephemeral file on Render) and SL/TP updates never reached the database at
all. Worse, the TP1 partial close wrote exit_time/pnl into the positions
row, so the final close OVERWROTE the partial's profit - per-trade stats
were wrong.

Covered here:
  - every open writes a ledger row (trade_uid + immutable entry price)
  - every SL/TP update is mirrored to the DB + trade_events audit trail
  - TP1 partial ACCUMULATES realized_pnl and keeps status='open'
  - the final close stores the WHOLE-trade result (partials + final chunk)
  - daily_stats pnl_delta is no longer always zero
  - startup restore recovers positions wiped from JSON (both directions)
  - autopsy lines + AI post-mortem (sanitized, optional)
"""
import sys
import json
from datetime import timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
import src.risk.manager as manager_module
from src.db.database import Database
from src.risk.manager import RiskManager, _new_trade_uid, _primary_strategy
from src.utils.helpers import now_utc


@pytest.fixture()
def ledger_db(tmp_path):
    """Real SQLite ledger isolated from data/bot_stats.db."""
    return Database(db_url=None, db_path=tmp_path / "ledger.db")


@pytest.fixture()
def rm(tmp_path, monkeypatch, ledger_db):
    """RiskManager wired to a REAL ledger DB + isolated JSON files."""
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
    return RiskManager(capital=10000)


def _stub_lot_rounding(monkeypatch):
    """open_paper_position consults Binance LOT_SIZE - keep it offline."""
    from src.core.binance_client import binance_client
    monkeypatch.setattr(binance_client, "round_quantity_to_lot",
                        lambda symbol, qty: round(qty, 8))


def make_rec(symbol="TESTUSDT", price=100.0, sl=98.0, tp=104.0, tp2=110.0):
    return {
        "symbol": symbol, "direction": "bullish", "current_price": price,
        "stop_loss": sl, "take_profit": tp, "take_profit_2": tp2,
        "confidence": 70.0, "expected_rise_pct": 4.0,
        "risk_reward_ratio": 2.0, "harmony": 0.8,
        "signals": [{"strategy": "triple_confluence_trend",
                     "direction": "bullish", "score": 4.0,
                     "confidence": 0.7}],
    }


def _open_one(rm, monkeypatch, **kw):
    _stub_lot_rounding(monkeypatch)
    res = rm.open_position(make_rec(**kw))
    assert res["status"] == "opened", res
    return res["position"]


# ============================================
# Open -> ledger row with immutable entry price
# ============================================

def test_uid_format_and_strategy_pick():
    uid = _new_trade_uid()
    assert uid.startswith("TRD-") and len(uid) > 10
    assert _new_trade_uid() != uid  # uniqueness
    assert _primary_strategy(make_rec()) == "triple_confluence_trend"


def test_open_paper_position_writes_ledger_row(rm, monkeypatch):
    pos = _open_one(rm, monkeypatch)
    assert pos["trade_uid"]
    db = manager_module.db
    open_rows = db.get_open_positions()
    assert len(open_rows) == 1
    row = open_rows[0]
    assert row["trade_uid"] == pos["trade_uid"]
    assert row["status"] == "open"
    assert row["entry_price"] == pytest.approx(100.0)
    assert row["initial_sl"] == pytest.approx(98.0)
    assert row["initial_tp"] == pytest.approx(104.0)
    assert row["tp2"] == pytest.approx(110.0)
    assert row["strategy"] == "triple_confluence_trend"
    # audit trail opens with an OPEN event
    evs = db.get_trade_events(pos["trade_uid"])
    assert [e["event_type"] for e in evs] == ["OPEN"]


# ============================================
# SL/TP updates -> DB mirror + audit events
# ============================================

def test_update_position_risk_persists_levels_and_event(rm, monkeypatch):
    _open_one(rm, monkeypatch)
    res = rm.update_position_risk(0, 101.0, new_sl=100.5,
                                  reason="Break-even lock")
    assert res["status"] == "updated"
    row = manager_module.db.get_open_positions()[0]
    assert row["stop_loss"] == pytest.approx(100.5)
    evs = manager_module.db.get_trade_events(row["trade_uid"])
    types = [e["event_type"] for e in evs]
    assert "RISK_UPDATE" in types
    ev = [e for e in evs if e["event_type"] == "RISK_UPDATE"][0]
    assert ev["new_sl"] == pytest.approx(100.5)
    assert ev["old_sl"] == pytest.approx(98.0)
    assert "Break-even" in ev["reason"]


def test_tp1_promotion_persists_tp1_levels_event(rm, monkeypatch):
    _open_one(rm, monkeypatch)
    results = rm.check_open_positions({"TESTUSDT": 105.0})
    assert any(r.get("status") == "partial" for r in results)
    row = manager_module.db.get_open_positions()[0]
    assert row["tp1_taken"] == 1
    assert row["stop_loss"] == pytest.approx(
        100.0 * (1 + settings.TP1_FEE_BUFFER_PCT / 100))
    assert row["take_profit"] == pytest.approx(110.0)  # promoted to TP2
    types = [e["event_type"]
             for e in manager_module.db.get_trade_events(row["trade_uid"])]
    assert "TP1_PARTIAL" in types
    assert "TP1_LEVELS" in types


# ============================================
# Partial close keeps the trade OPEN + accumulates
# ============================================

def test_partial_close_accumulates_realized_and_stays_open(rm, monkeypatch):
    pos = _open_one(rm, monkeypatch)
    orig_notional = pos["notional_usd"]  # captured BEFORE the partial
    results = rm.check_open_positions({"TESTUSDT": 105.0})
    partial = [r for r in results if r.get("status") == "partial"][0]
    row = manager_module.db.get_open_positions()[0]
    # THE FIX: partial close must NOT mark the trade closed
    assert row["status"] == "open"
    assert row["exit_time"] is None
    assert row["realized_pnl"] == pytest.approx(partial["pnl"])
    assert row["partial_count"] == 1
    # remaining notional synced to the ledger
    assert row["notional_usd"] == pytest.approx(orig_notional / 2)
    # in-memory position carries the realized total for the final close
    assert rm.open_positions[0]["realized_pnl"] == pytest.approx(partial["pnl"])


# ============================================
# Full close -> WHOLE-trade result (partials + final)
# ============================================

def test_full_close_records_whole_trade_total(rm, monkeypatch):
    _open_one(rm, monkeypatch)
    rm.check_open_positions({"TESTUSDT": 105.0})       # TP1 partial
    closed = rm.check_open_positions({"TESTUSDT": 111.0})  # TP2 hit
    fulls = [r for r in closed if r.get("status") == "closed"]
    assert len(fulls) == 1
    c = fulls[0]
    assert c["total_pnl"] == pytest.approx(c["pnl"] + c["realized_pnl"])
    db = manager_module.db
    rows = db.get_positions_history(closed_only=True, limit=10)
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == "closed"
    assert row["total_pnl"] == pytest.approx(c["total_pnl"])
    assert row["partial_count"] == 1
    # ledger is empty of open positions now
    assert db.get_open_positions() == []
    evs = db.get_trade_events(row["trade_uid"])
    assert evs[-1]["event_type"] == "CLOSE"
    # duration + updates count recorded for the autopsy
    assert c["duration_hours"] is not None
    assert c["partials_count"] == 1


def test_daily_stats_pnl_delta_no_longer_zero(rm, monkeypatch):
    _open_one(rm, monkeypatch)
    rm.check_open_positions({"TESTUSDT": 105.0})
    rm.check_open_positions({"TESTUSDT": 111.0})
    today = now_utc().strftime("%Y-%m-%d")
    rows = manager_module.db.get_daily_stats(limit=1)
    assert rows and rows[0]["date"] == today
    # JSON daily pnl == DB daily pnl (the old code always wrote 0)
    json_pnl = rm.daily_stats[today]["pnl"]
    assert rows[0]["pnl"] == pytest.approx(json_pnl)
    assert json_pnl > 0


def test_legacy_close_fallback_by_symbol_entry_time(rm, monkeypatch):
    """Pre-v5.11 row without trade_uid still closes via (symbol, entry_time)."""
    legacy = {
        "symbol": "OLDUSDT", "direction": "bullish", "entry_price": 100.0,
        "stop_loss": 98.0, "take_profit": 104.0, "size": 0.1,
        "notional_usd": 10.0, "entry_fee": 0.01,
        "entry_time": (now_utc() - timedelta(hours=1)).isoformat(),
        "status": "open",
    }
    manager_module.db.log_position_opened(legacy)  # row with NULL trade_uid
    rm.open_positions = [dict(legacy)]             # legacy in-memory position
    res = rm.close_position(0, 99.0, "Stop Loss Hit")
    assert res["status"] == "closed"
    row = manager_module.db.get_positions_history(closed_only=True)[0]
    assert row["status"] == "closed"
    assert row["close_reason"] == "Stop Loss Hit"


# ============================================
# Restore: the DB is the source of truth
# ============================================

def test_restore_recovers_position_wiped_from_json(rm, tmp_path,
                                                    monkeypatch):
    pos = _open_one(rm, monkeypatch)
    uid = pos["trade_uid"]
    entry = pos["entry_price"]
    # simulate the Render redeploy: the JSON file is gone, DB survives
    (tmp_path / "open_positions.json").write_text("[]")
    rm2 = RiskManager(capital=10000)
    assert len(rm2.open_positions) == 1
    got = rm2.open_positions[0]
    assert got["trade_uid"] == uid
    assert got["entry_price"] == pytest.approx(entry)
    assert got["stop_loss"] == pytest.approx(98.0)
    assert got["restored_from_db"] is True


def test_restore_backfills_json_positions_missing_from_db(rm, monkeypatch,
                                                          ledger_db,
                                                          tmp_path):
    _open_one(rm, monkeypatch)
    # simulate a fresh/rotated DB: rewrite the ledger file with empty tables
    fresh = Database(db_url=None, db_path=tmp_path / "fresh.db")
    monkeypatch.setattr(manager_module, "db", fresh)
    rm2 = RiskManager(capital=10000)  # same JSON, empty DB
    assert len(rm2.open_positions) == 1
    rows = fresh.get_open_positions()
    assert len(rows) == 1
    assert rows[0]["trade_uid"] == rm2.open_positions[0]["trade_uid"]
    assert rows[0]["entry_price"] == pytest.approx(100.0)


def test_restore_assigns_uid_to_legacy_positions(rm, monkeypatch, ledger_db):
    legacy = {
        "symbol": "OLDUSDT", "direction": "bullish", "entry_price": 5.0,
        "stop_loss": 4.8, "take_profit": 5.6, "size": 2.0,
        "notional_usd": 10.0, "entry_fee": 0.01,
        "entry_time": (now_utc() - timedelta(hours=2)).isoformat(),
        "status": "open",
    }
    rm.open_positions = [dict(legacy)]
    rm._restore_from_db()
    # legacy position got a uid + a ledger row
    assert rm.open_positions[0]["trade_uid"]
    rows = ledger_db.get_open_positions()
    assert len(rows) == 1
    assert rows[0]["symbol"] == "OLDUSDT"
    assert rows[0]["trade_uid"] == rm.open_positions[0]["trade_uid"]


def test_restore_disabled_leaves_everything_alone(rm, tmp_path, monkeypatch):
    _open_one(rm, monkeypatch)
    monkeypatch.setattr(settings, "TRADE_LEDGER_RESTORE", False)
    (tmp_path / "open_positions.json").write_text("[]")
    rm2 = RiskManager(capital=10000)  # restore gated off -> nothing recovered
    assert rm2.open_positions == []


# ============================================
# Dashboard P&L: whole-trade view after TP1
# ============================================

def test_positions_with_pnl_includes_realized_total(rm, monkeypatch):
    _open_one(rm, monkeypatch)
    rm.check_open_positions({"TESTUSDT": 105.0})  # bank TP1 (~+2.4% on half)
    view = rm.get_positions_with_pnl({"TESTUSDT": 105.0})[0]
    assert view["realized_pnl"] > 0
    assert view["total_pnl"] == pytest.approx(
        view["realized_pnl"] + view["current_pnl"])
    assert view["total_pnl_pct"] > 0


# ============================================
# Performance analytics + equity curve
# ============================================

def test_performance_stats_math(rm, monkeypatch):
    _open_one(rm, monkeypatch)
    rm.check_open_positions({"TESTUSDT": 105.0})
    rm.check_open_positions({"TESTUSDT": 111.0})   # winner
    _open_one(rm, monkeypatch, symbol="LOSERUSDT")
    rm.close_position(0, 97.0, "Stop Loss Hit")    # loser
    perf = manager_module.db.get_performance_stats(days=90)
    assert perf["closed_trades"] == 2
    assert perf["wins"] == 1 and perf["losses"] == 1
    assert perf["win_rate"] == pytest.approx(50.0)
    assert perf["net_pnl"] > 0                     # TP2 win dwarfs the SL loss
    assert perf["profit_factor"] > 1
    assert perf["avg_hold_hours"] >= 0
    curve = manager_module.db.get_equity_curve()
    assert len(curve) == 2
    assert curve[-1]["cum_pnl"] == pytest.approx(
        curve[0]["cum_pnl"] + curve[1]["pnl"])


# ============================================
# Autopsy card + AI post-mortem
# ============================================

def test_autopsy_lines_include_whole_trade_and_mfe():
    from src.core.cycle import _autopsy_lines
    lines = _autopsy_lines({
        "pnl": 1.2, "pnl_pct": 2.4, "total_pnl": 3.7, "total_pnl_pct": 7.4,
        "mfe_pct": 9.0, "mae_pct": 1.1, "capture_efficiency": 50.0,
        "duration_hours": 5.25, "partials_count": 1, "risk_updates_count": 3,
    })
    text = "\n".join(lines)
    assert "إجمالي الصفقة" in text and "+3.70" in text
    assert "+9.00%" in text and "-1.10%" in text
    assert "50%" in text and "5.2" in text
    # no autopsy line when total == chunk (no partials)
    assert _autopsy_lines({"pnl": 1.0, "total_pnl": 1.0}) == []


def test_ai_postmortem_disabled_and_sanitized(monkeypatch):
    from src.ai.llm_advisor import LLMAdvisor, ai_advisor
    # no API key in the test env -> disabled -> None (never raises)
    if not ai_advisor.enabled:
        assert ai_advisor.trade_postmortem({"symbol": "X"}) is None
    # enabled advisor + markdown-spewing model -> sanitized plain text
    adv = LLMAdvisor.__new__(LLMAdvisor)
    adv.enabled = True
    adv.model = "test"
    monkeypatch.setattr(adv, "_chat",
                        lambda *a, **k: "*ربح* جيد؛ `الوقف` عمل #حسناً")
    out = adv.trade_postmortem({"symbol": "SOLUSDT", "total_pnl_pct": 2.0})
    assert out is not None
    for ch in "*`#;":
        assert ch not in out
