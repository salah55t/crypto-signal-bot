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
