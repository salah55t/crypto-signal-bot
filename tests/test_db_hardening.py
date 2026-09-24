"""v5.1 DB hardening: Postgres init retries + no silent SQLite fallback.

Regression context (2026-09-24): when Postgres init failed on a Render
cold start, the old code silently switched to a fresh ephemeral SQLite
file. All positions/stats written there were wiped on the next deploy and
the dashboard showed no prices (positions list empty).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

import src.db.database as dbmod
from src.db.database import Database


def _pg(tmp_path):
    return Database(db_url="postgresql://user:pw@fake-host:5432/db",
                    db_path=tmp_path / "x.db")


def test_postgres_failure_retries_and_stays_postgres(tmp_path, monkeypatch):
    d = _pg(tmp_path)
    calls = {"n": 0}

    def failing_connect_raw():
        calls["n"] += 1
        raise ConnectionError("cold db")

    monkeypatch.setattr(d, "_connect_raw", failing_connect_raw)
    monkeypatch.setattr(dbmod.time, "sleep", lambda s: None)
    d._ensure_init()
    assert calls["n"] == 3                       # 3 attempts
    assert d.use_postgres is True                # NO silent sqlite fallback
    assert d._initialized is False               # stays uninitialized
    assert d._next_retry > 0                     # lazy retry scheduled


def test_postgres_lazy_retry_succeeds_on_wake(tmp_path, monkeypatch):
    d = _pg(tmp_path)
    state = {"fail": True}

    def flaky_connect_raw():
        if state["fail"]:
            raise ConnectionError("cold db")
        class _FakeConn:
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False
            def cursor(self):
                return self
            def execute(self, *a):
                pass
            def commit(self):
                pass
        return _FakeConn()

    monkeypatch.setattr(d, "_connect_raw", flaky_connect_raw)
    monkeypatch.setattr(dbmod.time, "sleep", lambda s: None)
    d._ensure_init()
    assert d._initialized is False

    state["fail"] = False          # "db woke up"
    d._next_retry = 0.0            # retry due now
    d._ensure_init()
    assert d._initialized is True  # schema applied, stays postgres


def test_sqlite_init_still_works(tmp_path):
    d = Database(db_url=None, db_path=tmp_path / "local.db")
    assert d.use_postgres is False
    assert d._initialized is True
    d.log_run("2026-09-24T00:00:00", 1.0, 1, 0, 0, 0, "paper", None)
    rows = d.get_daily_stats(limit=1)
    assert isinstance(rows, list)


def test_strategy_performance_postgres_casts_text_timestamp(tmp_path, monkeypatch):
    """Regression (2026-09-24): recommendations.timestamp is TEXT in Postgres
    (schema is shared with SQLite). Comparing it to NOW() - INTERVAL without
    an explicit cast failed with "operator does not exist: text >= timestamp
    with time zone", so /api/stats/strategies returned {"error": ...}."""
    d = _pg(tmp_path)
    captured = {}

    class _FakeCur:
        description = [("strategy_name",), ("total_signals",)]

        def execute(self, sql, params=None):
            captured["sql"] = sql
            captured["params"] = params

        def fetchall(self):
            return [("v5_confluence", 7)]

    class _FakeConn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def cursor(self):
            return _FakeCur()

    d._initialized = True                       # skip schema, exercise query only
    monkeypatch.setattr(d, "_connect", lambda: _FakeConn())

    rows = d.get_strategy_performance(days=30)
    assert rows == [{"strategy_name": "v5_confluence", "total_signals": 7}]
    # The TEXT timestamp column MUST be cast before comparing with timestamptz
    assert "recommendations.timestamp::timestamptz >= NOW() - INTERVAL" in captured["sql"]
    assert captured["params"] == ("30 days",)


def test_strategy_performance_sqlite_end_to_end(tmp_path):
    """SQLite branch of the same query: real temp DB, insert run +
    recommendation + strategy signal, verify aggregation works."""
    d = Database(db_url=None, db_path=tmp_path / "local.db")
    run_id = d.log_run("2026-09-24T03:00:00", 2.0, 5, 1, 1, 0, "paper", None)
    d.log_recommendation(run_id, {
        "analyzed_at": "2026-09-24T03:00:05+00:00",
        "symbol": "BTCUSDT",
        "direction": "bullish",
        "confidence": 0.8,
        "weighted_score": 7.5,
        "signals": [{
            "strategy": "v5_confluence",
            "direction": "bullish",
            "score": 3.0,
            "confidence": 0.9,
            "reasons": "test",
        }],
    })
    rows = d.get_strategy_performance(days=30)
    assert len(rows) == 1
    assert rows[0]["strategy_name"] == "v5_confluence"
    assert rows[0]["total_signals"] == 1
    assert rows[0]["bullish_signals"] == 1
