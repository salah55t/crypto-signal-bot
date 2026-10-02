"""
v5.26 Critical Fixes - Regression Tests

Five production-critical defects found by a full code review, each with a
regression lock here:

  1. LIVE position sizing ignored the stop entirely (`risk_amount * 10`):
     a 0.5%-SL trade risked ~5x the intended 1% while a 10%-SL trade
     risked ~0.5x. Sizing is now fixed-fractional off the REAL SL distance
     (position_size_notional), regime-multiplied, capped at 20% capital.
  2. open_live_position called can_open_position() WITHOUT the symbol, so
     the per-symbol re-entry cooldown after a losing close was bypassed
     on the live path only (paper passed it - the asymmetry was the bug).
  3. RiskManager had NO lock: open_positions was mutated from four
     concurrent contexts (analysis cron, 1-min watcher, manual POST
     triggers, /api/reset-history) and index-based close_position could
     pop the WRONG position after another thread shifted the list.
     - every public state method now runs under an instance RLock
     - uid-addressed close/update variants resolve the index FRESH
     - the structural-exit loop in cycle.py iterates a snapshot and
       closes by uid (covered indirectly here via the uid contract)
  4. The public dashboard exposed unauthenticated POST /api/reset-history
     (wipes ALL history) / run-analysis / scan-bottoms, and used the
     invalid CORS pair allow_origins=["*"] + allow_credentials=True.
     - POST endpoints now fail closed without DASHBOARD_API_TOKEN
     - CORS is credentialless-wildcard by default, credentialed only for
       an explicit allow-list
  5. save_json wrote non-atomically from multiple threads (a SIGTERM
     mid-write corrupted open_positions.json, silently degrading to [] on
     next load) and nothing stopped the scheduler/WS feed on shutdown.
     - save_json is now atomic (unique tmp + os.replace, per-thread tmp
       names, fsync, cleanup on failure)
     - shutdown_event really stops the scheduler and the WS feed
"""
import sys
import json
import threading
import asyncio
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
import src.risk.manager as manager_module
from src.risk.manager import RiskManager
from src.db.database import Database
from src.utils.helpers import load_json, save_json


# ============================================================
# Fixtures
# ============================================================

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
    """open_paper/open_live consult Binance LOT_SIZE - keep it offline."""
    from src.core.binance_client import binance_client
    monkeypatch.setattr(binance_client, "round_quantity_to_lot",
                        lambda symbol, qty: round(qty, 8))


def _neutral_regime(monkeypatch):
    """Deterministic gates: no regime adjust, no size multiplier."""
    monkeypatch.setattr(RiskManager, "_regime_gates",
                        staticmethod(lambda: (68.0, 1.5)))
    monkeypatch.setattr(RiskManager, "_regime_size_multiplier",
                        staticmethod(lambda: 1.0))


def _valid_rec(symbol="TESTUSDT", entry=100.0, sl=90.0, tp=110.0):
    """A rec that passes validate_recommendation (conf 75, RR 2.0,
    harmony 0.8, coherent bullish geometry)."""
    return {
        "symbol": symbol,
        "direction": "bullish",
        "current_price": entry,
        "stop_loss": sl,
        "take_profit": tp,
        "take_profit_2": tp,
        "confidence": 75.0,
        "admission_confidence": 75.0,
        "expected_rise_pct": 2.0,
        "risk_reward_ratio": 2.0,
        "harmony": 0.8,
        "volatility_extreme": False,
        "dead_market": False,
        "decision": {},
        "signals": [{"strategy": "trend_pullback", "score": 60}],
        "strategy": "trend_pullback",
    }


# ============================================================
# FIX 1: SL-linked live sizing
# ============================================================

def test_sizing_hit_stop_loses_exactly_risk_per_trade(rm):
    """entry 100 / SL 90 (10% away): notional = 1% risk / 10% = $1000.
    Hitting the stop (plus fees aside) must lose ~RISK_PER_TRADE=1%."""
    notional = rm.position_size_notional(100.0, 90.0)
    assert notional == pytest.approx(1000.0, rel=1e-9)
    loss_at_stop_pct = (notional * 0.10) / rm.capital * 100
    assert loss_at_stop_pct == pytest.approx(1.0, rel=1e-6)


def test_sizing_capped_at_20pct_capital(rm):
    """A 0.5%-SL trade would want $20,000 (20x risk) - capped at 20%."""
    assert rm.position_size_notional(100.0, 99.5) == pytest.approx(2000.0)


def test_sizing_fallback_when_sl_unusable(rm):
    """Missing/invalid SL -> conservative 5x risk (assumes ~2% stop),
    still capped - never a crash, never the old unlinked 10x."""
    assert rm.position_size_notional(100.0, None) == pytest.approx(500.0)
    assert rm.position_size_notional(100.0, 0) == pytest.approx(500.0)
    assert rm.position_size_notional(None, 90.0) == pytest.approx(500.0)
    assert rm.position_size_notional(100.0, 100.0) == pytest.approx(500.0)


def test_sizing_regime_multiplier_applies(rm, monkeypatch):
    monkeypatch.setattr(RiskManager, "_regime_size_multiplier",
                        staticmethod(lambda: 0.5))
    assert rm.position_size_notional(100.0, 90.0) == pytest.approx(500.0)


def test_sizing_units_api_consistent_with_notional(rm):
    """position_size (base units) x entry == position_size_notional."""
    units = rm.position_size(100.0, 90.0)
    assert units * 100.0 == pytest.approx(rm.position_size_notional(100.0, 90.0))


def test_live_open_uses_sl_linked_notional(rm, monkeypatch):
    """End-to-end on the live path: the MARKET BUY quote must equal the
    SL-derived notional (old heuristic would have sent risk_amount*10)."""
    _stub_lot_rounding(monkeypatch)
    _neutral_regime(monkeypatch)
    monkeypatch.setattr(settings, "USE_PUBLIC_ONLY", False)
    monkeypatch.setattr(settings, "RUN_MODE", "live")

    from src.core.binance_client import binance_client
    buys = {}

    def fake_market_buy(symbol, quote_qty):
        buys["symbol"] = symbol
        buys["quote_qty"] = quote_qty
        return {"orderId": 1, "executedQty": "10",
                "cummulativeQuoteQty": str(quote_qty)}

    monkeypatch.setattr(binance_client, "place_market_buy", fake_market_buy)
    monkeypatch.setattr(binance_client, "get_symbol_filters",
                        lambda s: {"lot_size_step": 0.00000001,
                                   "tick_size": 0.00000001})
    monkeypatch.setattr(binance_client, "place_oco_sell",
                        lambda **kw: {"orderListId": 9})

    rec = _valid_rec(entry=100.0, sl=90.0)  # 10% SL -> $1000 notional
    res = rm.open_live_position(rec)
    assert res["status"] == "opened", res
    assert buys["quote_qty"] == pytest.approx(1000.0)
    assert res["position"]["notional_usd"] == pytest.approx(1000.0)


def test_live_open_rejects_below_binance_minimum(rm, monkeypatch):
    """Tiny capital + wide fallback -> notional < $10 must still reject."""
    _neutral_regime(monkeypatch)
    monkeypatch.setattr(settings, "USE_PUBLIC_ONLY", False)
    monkeypatch.setattr(settings, "RUN_MODE", "live")
    tiny = RiskManager(capital=50)  # risk 0.5 -> fallback notional 2.5
    tiny._lock = rm._lock  # unused; keep the instance self-consistent
    res = tiny.open_live_position(_valid_rec())
    assert res["status"] == "rejected"
    assert "below Binance minimum" in res["reasons"][0]


# ============================================================
# FIX 2: live path honors the per-symbol re-entry cooldown
# ============================================================

def test_live_path_blocks_reentry_after_loss(rm, monkeypatch):
    """A symbol that just stopped out is NOT re-buyable live (the old live
    call dropped the symbol -> cooldown silently bypassed with real money)."""
    _stub_lot_rounding(monkeypatch)
    _neutral_regime(monkeypatch)
    monkeypatch.setattr(settings, "USE_PUBLIC_ONLY", False)
    monkeypatch.setattr(settings, "RUN_MODE", "live")
    from datetime import timedelta
    from src.utils.helpers import now_utc

    from src.core.binance_client import binance_client
    monkeypatch.setattr(binance_client, "place_market_buy",
                        lambda s, q: (_ for _ in ()).throw(AssertionError(
                            "no order may be placed during cooldown")))
    monkeypatch.setattr(binance_client, "get_symbol_filters",
                        lambda s: {"lot_size_step": 0.00000001,
                                   "tick_size": 0.00000001})
    monkeypatch.setattr(binance_client, "place_oco_sell",
                        lambda **kw: {"orderListId": 9})

    rm._reentry_block["TESTUSDT"] = now_utc() + timedelta(hours=2)
    res = rm.open_live_position(_valid_rec())
    assert res["status"] == "rejected"
    assert res["reasons"] == ["Risk limits reached"]


def test_paper_path_still_blocks_reentry(rm, monkeypatch):
    """Parity guard: paper keeps passing the symbol (pre-existing behavior)."""
    _stub_lot_rounding(monkeypatch)
    _neutral_regime(monkeypatch)
    from datetime import timedelta
    from src.utils.helpers import now_utc
    rm._reentry_block["TESTUSDT"] = now_utc() + timedelta(hours=2)
    res = rm.open_paper_position(_valid_rec())
    assert res["status"] == "rejected"
    assert res["reasons"] == ["Risk limits reached"]


# ============================================================
# FIX 3: lock + uid-addressed close/update
# ============================================================

def _open_n(rm, monkeypatch, n):
    _stub_lot_rounding(monkeypatch)
    _neutral_regime(monkeypatch)
    # lifts caps so a batch open is possible in tests
    monkeypatch.setattr(settings, "MAX_OPEN_POSITIONS", 64)
    monkeypatch.setattr(settings, "MAX_TRADES_PER_DAY", 200)
    uids = []
    for i in range(n):
        res = rm.open_paper_position(
            _valid_rec(symbol=f"SYM{i}USDT", entry=100.0, sl=95.0, tp=110.0))
        assert res["status"] == "opened"
        uids.append(res["position"]["trade_uid"])
    return uids


def test_clear_all_state_wipes_everything(rm, monkeypatch):
    uids = _open_n(rm, monkeypatch, 2)
    rm.daily_stats[rm._today_key()]["losses"] = 3
    rm._loss_streak = 3
    rm._reentry_block["FOO"] = None
    rm.clear_all_state()
    assert rm.open_positions == []
    assert rm.daily_stats == {}
    assert rm._loss_streak == 0
    assert rm._loss_pause_until is None
    assert rm._reentry_block == {}
    # stats re-create themselves zeroed on next touch
    rm._ensure_today_stats()
    assert rm.daily_stats[rm._today_key()]["trades_opened"] == 0


def test_close_position_by_uid_closes_the_right_one(rm, monkeypatch):
    uids = _open_n(rm, monkeypatch, 3)
    symbols = [p["trade_uid"] for p in rm.open_positions]
    target_uid = uids[1]
    closed = rm.close_position_by_uid(target_uid, 99.0, "unit test")
    assert closed.get("status") == "closed"
    assert closed["trade_uid"] == target_uid
    remaining = [p["trade_uid"] for p in rm.open_positions]
    assert target_uid not in remaining
    assert len(remaining) == 2


def test_close_position_by_uid_stale_identity_is_safe(rm, monkeypatch):
    """Closing an already-closed uid returns an ERROR, never a neighbor."""
    uids = _open_n(rm, monkeypatch, 2)
    first = rm.close_position_by_uid(uids[0], 99.0, "first close")
    assert first["status"] == "closed"
    again = rm.close_position_by_uid(uids[0], 99.0, "stale repeat")
    assert again["status"] == "error"
    assert "not found" in again["reason"]
    # the survivor is untouched
    assert [p["trade_uid"] for p in rm.open_positions] == [uids[1]]


def test_update_position_risk_by_uid(rm, monkeypatch):
    uids = _open_n(rm, monkeypatch, 2)
    res = rm.update_position_risk_by_uid(uids[1], 101.0, 97.0, None, "tighten")
    assert res["status"] == "updated"
    assert rm.open_positions[-1]["stop_loss"] == 97.0
    gone = rm.update_position_risk_by_uid("TRD-NOPE", 101.0, 97.0)
    assert gone["status"] == "error"


def test_lock_serializes_concurrent_close_and_watch(rm, monkeypatch):
    """5 threads close distinct uids while 5 threads run the watcher pass:
    every uid closes exactly once, the RIGHT position leaves the list, and
    no thread sees an inconsistent index (the pre-v5.26 race)."""
    uids = _open_n(rm, monkeypatch, 20)
    safe_prices = {f"SYM{i}USDT": 100.0 for i in range(20)}  # no SL/TP hit
    errors = []
    closed_uids = []
    guard = threading.Lock()
    to_close = uids[:10]

    def closer(uid):
        try:
            res = rm.close_position_by_uid(uid, 99.5, "concurrent")
            with guard:
                closed_uids.append((uid, res.get("status")))
        except Exception as e:  # pragma: no cover - race evidence
            with guard:
                errors.append(repr(e))

    def watcher():
        try:
            for _ in range(20):
                rm.check_open_positions(safe_prices)
        except Exception as e:  # pragma: no cover
            with guard:
                errors.append(repr(e))

    threads = [threading.Thread(target=closer, args=(u,)) for u in to_close]
    threads += [threading.Thread(target=watcher) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert errors == []
    statuses = {s for _, s in closed_uids}
    assert statuses == {"closed"}
    assert sorted(u for u, _ in closed_uids) == sorted(to_close)
    remaining = {p["trade_uid"] for p in rm.open_positions}
    assert remaining == set(uids[10:])


# ============================================================
# FIX 4: control-plane auth + CORS
# ============================================================

def test_reset_history_requires_token(monkeypatch, tmp_path):
    """Fail CLOSED: without DASHBOARD_API_TOKEN the destructive endpoint
    is disabled with 403 (it used to wipe ALL history for anyone)."""
    from src.web import app as app_module
    from fastapi.testclient import TestClient
    client = TestClient(app_module.app)
    monkeypatch.setattr(settings, "DASHBOARD_API_TOKEN", "")
    r = client.post("/api/reset-history")
    assert r.status_code == 403
    assert "DASHBOARD_API_TOKEN" in r.json()["detail"]


def test_reset_history_rejects_wrong_token(monkeypatch):
    from src.web import app as app_module
    from fastapi.testclient import TestClient
    client = TestClient(app_module.app)
    monkeypatch.setattr(settings, "DASHBOARD_API_TOKEN", "tok-abc")
    r = client.post("/api/reset-history")
    assert r.status_code == 401
    r = client.post("/api/reset-history",
                    headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401


def test_reset_history_accepts_valid_token_and_clears_state(
        monkeypatch, tmp_path):
    from src.web import app as app_module
    from fastapi.testclient import TestClient
    from src.db import database as db_module
    from src.risk import manager as risk_module

    # isolate: DATA_DIR + json files + DB reset + in-memory clear recorder
    for name in ["open_positions.json", "closed_trades.json",
                 "daily_stats.json", "recommendations.json",
                 "bottom_candidates.json"]:
        (tmp_path / name).write_text("{}")
    monkeypatch.setattr(app_module, "DATA_DIR", tmp_path)
    monkeypatch.setattr(app_module, "POSITIONS_FILE", tmp_path / "open_positions.json")
    monkeypatch.setattr(app_module, "STATS_FILE", tmp_path / "daily_stats.json")
    monkeypatch.setattr(app_module, "RECOMMENDATIONS_FILE",
                        tmp_path / "recommendations.json")
    calls = {"db": 0, "clear": 0}
    monkeypatch.setattr(db_module.db, "reset_all_history",
                        lambda: {"status": calls.__setitem__("db", 1) or "ok"})
    real_clear = risk_module.risk_manager.clear_all_state

    def recorder():
        calls["clear"] += 1
        return real_clear()

    monkeypatch.setattr(risk_module.risk_manager, "clear_all_state", recorder)
    monkeypatch.setattr(settings, "DASHBOARD_API_TOKEN", "tok-abc")

    client = TestClient(app_module.app)
    r = client.post("/api/reset-history",
                    headers={"X-Auth-Token": "tok-abc"})
    assert r.status_code == 200
    assert r.json()["status"] == "success"
    assert calls == {"db": 1, "clear": 1}
    # open_positions.json is DELETED then re-created as an empty store
    assert load_json(tmp_path / "open_positions.json", default=None) == []
    assert load_json(tmp_path / "daily_stats.json", default=None) == {}


def test_run_analysis_and_scan_bottoms_gated(monkeypatch):
    from src.web import app as app_module
    from fastapi.testclient import TestClient
    client = TestClient(app_module.app)
    monkeypatch.setattr(settings, "DASHBOARD_API_TOKEN", "tok-abc")
    # no token -> 401 (endpoint would burn REST weight / start cycles)
    assert client.post("/api/run-analysis").status_code == 401
    assert client.post("/api/scan-bottoms").status_code == 401
    # correct token -> started; cycle/scan bodies are stubbed offline
    monkeypatch.setattr(app_module, "run_bot_cycle_sync", lambda: None)
    # NOTE: src/analysis/__init__ re-exports the bottom_scanner INSTANCE
    # (same trap as data_fetcher) - patch the real module's instance.
    import importlib
    bs_mod = importlib.import_module("src.analysis.bottom_scanner")
    monkeypatch.setattr(bs_mod.bottom_scanner, "scan", lambda **kw: None)
    r1 = client.post("/api/run-analysis",
                     headers={"Authorization": "Bearer tok-abc"})
    r2 = client.post("/api/scan-bottoms", headers={"X-Auth-Token": "tok-abc"})
    assert r1.status_code == 200 and r1.json()["status"] == "started"
    assert r2.status_code == 200 and r2.json()["status"] == "started"


def test_cors_default_is_credentialless_wildcard():
    """The invalid `*` + credentials=True pair is gone: preflight succeeds
    for reads but never grants credentials cross-origin."""
    from src.web import app as app_module
    from fastapi.testclient import TestClient
    client = TestClient(app_module.app)
    r = client.options("/api/health", headers={
        "Origin": "https://attacker.example",
        "Access-Control-Request-Method": "GET",
    })
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") in ("*", "https://attacker.example")
    assert r.headers.get("access-control-allow-credentials") != "true"


# ============================================================
# FIX 5: atomic save_json + real shutdown
# ============================================================

def test_save_json_roundtrip_and_no_tmp_leftovers(tmp_path):
    target = tmp_path / "state.json"
    save_json({"a": 1, "b": [1, 2, 3]}, target)
    assert load_json(target, default=None) == {"a": 1, "b": [1, 2, 3]}
    assert list(tmp_path.glob(".*tmp*")) == []


def test_save_json_failure_keeps_previous_file_intact(tmp_path):
    target = tmp_path / "state.json"
    save_json({"version": "good"}, target)
    # A DIRECTORY at the target path makes os.replace fail AFTER the tmp
    # write - the error must be swallowed (logged), the tmp cleaned up and
    # every other file left untouched.
    blocker = tmp_path / "blocked.json"
    blocker.mkdir()
    save_json({"nope": True}, blocker)
    assert load_json(target, default=None) == {"version": "good"}
    assert blocker.is_dir()  # was never replaced by a file
    assert list(tmp_path.glob(".*tmp*")) == []


def test_save_json_recovers_from_corrupt_target(tmp_path):
    target = tmp_path / "state.json"
    target.write_text('{"truncated": tru')  # killed-mid-write debris
    save_json({"healed": True}, target)
    assert load_json(target, default=None) == {"healed": True}


def test_save_json_concurrent_writers_never_corrupt(tmp_path):
    """8 threads x 40 saves on ONE path (the pre-v5.26 production pattern):
    the file must always be valid JSON and no tmp debris may survive."""
    target = tmp_path / "open_positions.json"
    errors = []

    def writer(tid):
        try:
            for i in range(40):
                save_json({"writer": tid, "i": i,
                           "payload": [tid, i, tid * 10 + i]}, target)
        except Exception as e:  # pragma: no cover
            errors.append(repr(e))

    threads = [threading.Thread(target=writer, args=(t,)) for t in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert errors == []
    data = load_json(target, default=None)
    assert isinstance(data, dict) and "writer" in data
    assert list(tmp_path.glob(".*tmp*")) == []


def test_shutdown_event_stops_scheduler_and_ws_feed(monkeypatch):
    """v5.26: shutdown is no longer a log line - jobs and the WS feed are
    really stopped (uvicorn SIGTERM -> shutdown events)."""
    from src.web import app as app_module
    from src.core.ws_feed import ws_feed
    stopped = {"sched": False, "ws": False}

    class FakeSched:
        def shutdown(self, wait=False):
            stopped["sched"] = True

    monkeypatch.setattr(app_module, "_scheduler", FakeSched(), raising=False)

    real_stop = ws_feed.stop
    monkeypatch.setattr(ws_feed, "stop",
                        lambda: (stopped.__setitem__("ws", True),
                                 real_stop())[1])

    asyncio.run(app_module.shutdown_event())
    assert stopped["sched"] is True
    assert stopped["ws"] is True


def test_shutdown_event_survives_missing_scheduler():
    """Fresh process / failed startup: shutdown must not raise."""
    from src.web import app as app_module
    monkeypatch_none = None
    import src.web.app as _app
    _app._scheduler = None
    asyncio.run(_app.shutdown_event())  # must not raise
