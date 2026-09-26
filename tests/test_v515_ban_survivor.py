"""
v5.15 ban-survivor tests.

Production incident (2026-09-26): a Render deploy restarted the process
during a still-active Binance 418 IP ban. The in-memory cooldown died with
the old process, so the fresh one immediately fired /ticker/24hr (weight 80)
into the banned IP -> "Failed to fetch USDT pairs: Binance 418" -> fresh
2476s cooldown -> dynamic universe silently shrank to the static list.

Fixes covered here:
  1. rate_limiter persists the cooldown to data/rate_state.json (epoch) and
     a NEW limiter instance restores it at construction (restart-safe).
  2. Expired / corrupt state files are ignored (never crash the boot).
  3. Binance 418 honours the server Retry-After up to 24h (no 1h re-poke).
  4. 429 semantics unchanged (Retry-After + 2s, cap 3600).
  5. MarketAnalyzer.__init__ / refresh_symbols NEVER touch REST while a ban
     is active: disk-cached universe first, static coins.yaml second, and
     zero network weight spent.
  6. Successful dynamic fetches persist data/symbols_cache.json for later
     ban-safe boots; stale caches (older than SYMBOLS_CACHE_MAX_AGE_H) are
     rejected.

NOTE (mandatory project pattern): src.core / src.analysis re-export the
singleton under the module name, so tests MUST import modules with
importlib.import_module and patch attributes on the MODULE (not on re-
export aliases).
"""
import importlib
import json
import time
from pathlib import Path

import pytest


def _mod(name: str):
    return importlib.import_module(name)


# ----------------------------------------------------------------------
# Rate limiter persistence
# ----------------------------------------------------------------------
@pytest.fixture
def rl(tmp_path, monkeypatch):
    mod = _mod("src.core.rate_limiter")
    monkeypatch.setattr(mod, "STATE_FILE", tmp_path / "rate_state.json")
    limiter = mod.WeightedRateLimiter(budget_per_min=4500.0)
    return mod, limiter, tmp_path


def test_trigger_cooldown_persists_state_file(rl):
    mod, limiter, tmp_path = rl
    limiter.trigger_cooldown(600.0)
    f = tmp_path / "rate_state.json"
    assert f.exists()
    payload = json.loads(f.read_text())
    remaining = payload["cooldown_until_epoch"] - time.time()
    assert 590.0 <= remaining <= 600.0


def test_short_pressure_blips_are_not_persisted(rl):
    mod, limiter, tmp_path = rl
    limiter.trigger_cooldown(30.0)  # < RATE_STATE_MIN_S (60)
    assert not (tmp_path / "rate_state.json").exists()


def test_new_instance_restores_cooldown_from_disk(rl):
    import os
    mod, limiter, tmp_path = rl
    limiter.trigger_cooldown(900.0)
    # Simulate a REAL restart: the state file must look like it was written
    # by a DIFFERENT process (the pid guard keeps same-process semantics).
    f = tmp_path / "rate_state.json"
    payload = json.loads(f.read_text())
    payload["pid"] = os.getpid() + 424242
    f.write_text(json.dumps(payload))
    fresh = mod.WeightedRateLimiter(budget_per_min=4500.0)
    left = fresh.cooldown_remaining()
    assert 880.0 <= left <= 900.0
    assert fresh.in_cooldown()


def test_same_process_instances_do_not_restore(rl):
    """In ONE process the in-memory state is authoritative (pid guard)."""
    import os
    mod, limiter, tmp_path = rl
    limiter.trigger_cooldown(900.0)  # writes file with OUR pid
    fresh = mod.WeightedRateLimiter(budget_per_min=4500.0)
    assert fresh.cooldown_remaining() == 0.0  # not inherited within a process


def test_expired_state_file_is_ignored_and_removed(rl):
    mod, limiter, tmp_path = rl
    f = tmp_path / "rate_state.json"
    f.write_text(json.dumps({"cooldown_until_epoch": time.time() - 5.0}))
    fresh = mod.WeightedRateLimiter(budget_per_min=4500.0)
    assert fresh.cooldown_remaining() == 0.0
    assert not f.exists()  # stale file cleaned up


def test_corrupt_state_file_is_ignored(rl):
    mod, limiter, tmp_path = rl
    f = tmp_path / "rate_state.json"
    f.write_text("{not valid json!!")
    fresh = mod.WeightedRateLimiter(budget_per_min=4500.0)
    assert fresh.cooldown_remaining() == 0.0


def test_restored_ban_blocks_requests_without_network(rl):
    """A restored cooldown must gate acquire() exactly like a live one."""
    import os
    mod, limiter, tmp_path = rl
    limiter.trigger_cooldown(1200.0)
    f = tmp_path / "rate_state.json"
    payload = json.loads(f.read_text())
    payload["pid"] = os.getpid() + 999999  # simulate a genuine restart
    f.write_text(json.dumps(payload))
    fresh = mod.WeightedRateLimiter(budget_per_min=4500.0)
    ok = fresh.acquire(weight=2, timeout=0.5)
    assert ok is False  # refused fast - no 90s hang, no network


# ----------------------------------------------------------------------
# 418 / 429 Retry-After semantics (via client._get with a fake session)
# ----------------------------------------------------------------------
@pytest.fixture
def bc_mod(tmp_path, monkeypatch):
    mod = _mod("src.core.binance_client")
    import src.core.rate_limiter as rl_mod
    monkeypatch.setattr(
        "src.core.rate_limiter.STATE_FILE", tmp_path / "rate_state.json"
    )
    # The real singleton is shared across the whole suite - snapshot & reset
    # its cooldown so tests neither see nor leak ban state.
    monkeypatch.setattr(rl_mod.rate_limiter, "_cooldown_until", 0.0)
    yield mod


class _FakeResponse:
    def __init__(self, status, retry_after, headers=None):
        self.status_code = status
        self.text = "boom"
        self.headers = headers or {}
        if retry_after is not None:
            self.headers["Retry-After"] = str(retry_after)

    def raise_for_status(self):
        return None

    def json(self):
        return {}


def test_418_retry_after_9000_is_honoured_not_capped_at_3600(bc_mod, monkeypatch):
    import src.core.rate_limiter as rl_mod  # real module object
    client = bc_mod.binance_client
    monkeypatch.setattr(
        client.session, "get",
        lambda *a, **k: _FakeResponse(418, 9000), raising=True,
    )
    with pytest.raises(bc_mod.RateLimitError):
        client._get("/api/v3/ping")
    left = rl_mod.rate_limiter.cooldown_remaining()
    assert 8000.0 <= left <= 9000.0  # v5.15: was capped at 3600 before


def test_429_retry_after_semantics_unchanged(bc_mod, monkeypatch):
    import src.core.rate_limiter as rl_mod
    client = bc_mod.binance_client
    monkeypatch.setattr(
        client.session, "get",
        lambda *a, **k: _FakeResponse(429, 300), raising=True,
    )
    with pytest.raises(bc_mod.RateLimitError):
        client._get("/api/v3/ping")
    left = rl_mod.rate_limiter.cooldown_remaining()
    assert 290.0 <= left <= 303.0  # 300 + 2 grace, cap 3600 untouched


# ----------------------------------------------------------------------
# Analyzer: ban-safe symbol boot + refresh + disk cache
# ----------------------------------------------------------------------
@pytest.fixture
def az(tmp_path, monkeypatch):
    mod = _mod("src.analysis.analyzer")
    monkeypatch.setattr(mod, "SYMBOLS_CACHE_FILE", tmp_path / "symbols_cache.json")
    monkeypatch.setattr(mod.settings, "USE_ALL_USDT_PAIRS", True)
    monkeypatch.setattr(mod.settings, "MAX_SYMBOLS", 50)
    monkeypatch.setattr(mod.settings, "MIN_VOLUME_USDT", 1_000_000)
    return mod, tmp_path


def _mk_analyzer(az, monkeypatch, banned: bool):
    mod, tmp_path = az
    monkeypatch.setattr(mod, "_ban_active", lambda *a, **k: banned)
    return mod.MarketAnalyzer()


def test_init_during_ban_uses_disk_cache_no_network(az, monkeypatch):
    from src.utils.helpers import now_utc
    mod, tmp_path = az
    (tmp_path / "symbols_cache.json").write_text(json.dumps({
        "symbols": ["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        "saved_at": now_utc().isoformat(),
        "count": 3,
    }))
    calls = {"n": 0}

    def _boom():
        calls["n"] += 1
        raise AssertionError("network touched during ban!")

    from src.core.binance_client import binance_client
    monkeypatch.setattr(binance_client, "get_all_tickers", _boom)
    a = _mk_analyzer(az, monkeypatch, banned=True)
    assert calls["n"] == 0
    assert a.symbols == ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def test_init_during_ban_without_cache_falls_back_to_static_no_network(az, monkeypatch):
    mod, tmp_path = az
    static = list(mod.settings.load_coins())[:3]
    monkeypatch.setattr(mod.settings, "load_coins", lambda: static)

    def _boom():
        raise AssertionError("network touched during ban!")

    from src.core.binance_client import binance_client
    monkeypatch.setattr(binance_client, "get_all_tickers", _boom)
    a = _mk_analyzer(az, monkeypatch, banned=True)
    assert a.symbols == static  # no shrink to [] and no network


def test_successful_fetch_persists_symbols_cache(az, monkeypatch):
    mod, tmp_path = az
    tickers = [
        {"symbol": "BTCUSDT", "quoteVolume": "99999999"},
        {"symbol": "ETHUSDT", "quoteVolume": "50000000"},
        {"symbol": "EURUSDT", "quoteVolume": "99999999"},  # not a USDT quote pair? it is - keep
    ]
    from src.core.binance_client import binance_client
    monkeypatch.setattr(binance_client, "get_all_tickers", lambda: tickers)
    a = _mk_analyzer(az, monkeypatch, banned=False)
    assert "BTCUSDT" in a.symbols and "ETHUSDT" in a.symbols
    payload = json.loads((tmp_path / "symbols_cache.json").read_text())
    assert "BTCUSDT" in payload["symbols"]
    assert payload["count"] == len(a.symbols)


def test_refresh_symbols_during_ban_is_noop(az, monkeypatch):
    mod, tmp_path = az
    a = _mk_analyzer(az, monkeypatch, banned=True)
    a.symbols = ["KEEPUSDT"]
    a._symbols_ts = mod.now_utc()

    def _boom():
        raise AssertionError("refresh touched the network during a ban!")

    from src.core.binance_client import binance_client
    monkeypatch.setattr(binance_client, "get_all_tickers", _boom)
    a.refresh_symbols(force=True)
    assert a.symbols == ["KEEPUSDT"]  # unchanged
    assert a._symbols_ts is not None


def test_refresh_symbols_after_ban_fetches_normally(az, monkeypatch):
    mod, tmp_path = az
    a = _mk_analyzer(az, monkeypatch, banned=True)
    a.symbols = ["STALEUSDT"]
    tickers = [{"symbol": "FRESHUSDT", "quoteVolume": "88888888"}]
    from src.core.binance_client import binance_client
    monkeypatch.setattr(binance_client, "get_all_tickers", lambda: tickers)
    # ban lifted -> refresh must fetch
    monkeypatch.setattr(mod, "_ban_active", lambda *a, **k: False)
    a.refresh_symbols(force=True)
    assert a.symbols == ["FRESHUSDT"]


def test_stale_symbols_cache_is_rejected(az, monkeypatch):
    from datetime import datetime, timedelta, timezone
    from src.utils.helpers import now_utc
    mod, tmp_path = az
    old = now_utc() - timedelta(hours=200)  # > 168h default
    (tmp_path / "symbols_cache.json").write_text(json.dumps({
        "symbols": ["OLDUSDT"], "saved_at": old.isoformat(), "count": 1,
    }))
    static = ["STATIC1USDT"]
    monkeypatch.setattr(mod.settings, "load_coins", lambda: static)

    def _boom():
        raise AssertionError("network touched during ban!")

    from src.core.binance_client import binance_client
    monkeypatch.setattr(binance_client, "get_all_tickers", _boom)
    a = _mk_analyzer(az, monkeypatch, banned=True)
    assert a.symbols == static  # stale cache ignored, static fallback


def test_ban_active_mid_fetch_keeps_cache_without_error_spam(az, monkeypatch):
    """A 418 arriving mid-fetch falls back to cache quietly (no crash)."""
    from src.utils.helpers import now_utc
    mod, tmp_path = az
    (tmp_path / "symbols_cache.json").write_text(json.dumps({
        "symbols": ["CACHEDUSDT"],
        "saved_at": now_utc().isoformat(),
        "count": 1,
    }))
    ban = {"on": False}

    def _ban():
        return ban["on"]

    def _boom():
        ban["on"] = True  # ban went active while the request was in flight
        raise RuntimeError("Binance 418: Way too much request weight used")

    from src.core.binance_client import binance_client
    monkeypatch.setattr(binance_client, "get_all_tickers", _boom)
    monkeypatch.setattr(mod, "_ban_active", _ban)
    a = _mk_analyzer(az, monkeypatch, banned=False)
    assert a.symbols == ["CACHEDUSDT"]
