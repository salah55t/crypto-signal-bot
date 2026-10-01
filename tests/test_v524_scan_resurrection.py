"""v5.24 scan-resurrection: the bot that looked dead.

Production 2026-10-01 forensics: the process booted inside a restored
shared-IP cooldown (v5.15 rate_state.json), the WS kline streams created
1-bar cache entries for every universe symbol the moment they connected,
and the paced REST seeder's `_missing_keys()` saw those shallow keys as
"already cached" - so it never deepened anything. coverage() stayed 0.0
forever, every cron tick skipped, the recovery windows were burned by fat
REST bursts that earned fresh 429s (ONE request -> 732s Retry-After), and
`total_runs` / `last_run` stayed at zero: the dashboard showed a bot that
had NEVER scanned.

Fixes under test:
  1. ws_feed._missing_keys() is DEPTH-AWARE (shallow keys are seeder work)
     with failed-key (30-min) and short-history exclusion.
  2. ws_feed.servable_coverage() measures what get_cached() actually
     serves (CANDLE_LIMIT-deep + fresh) - the old 60-bar coverage()
     declared mid-depth caches "ready" that serve nothing.
  3. ws_feed.ensure_seed_worker() - lazy worker start from anywhere.
  4. cycle.rate_limit_gate(): recovery-window yield + stall-breaker +
     partial WS-only escape hatch after repeated skips.
  5. Settings contract + dashboard transparency fields.
"""
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pytest

import importlib
from config.settings import settings

wsmod = importlib.import_module("src.core.ws_feed")
cycmod = importlib.import_module("src.core.cycle")
WSKlineFeed = wsmod.WSKlineFeed

IVS = ("1h", "4h")


# ---------- helpers ----------

def _mk_feed(symbols, bars_per_key, fresh=True):
    """Feed whose every (symbol, interval) key holds `bars_per_key` bars."""
    f = WSKlineFeed()
    f._started = True
    f._universe = list(symbols)
    now = time.monotonic()
    for s in symbols:
        for iv in IVS:
            key = f._key(s, iv)
            f._bars[key] = [[0] * 12 for _ in range(bars_per_key)]
            f._last_event[key] = now if fresh else now - 9999
    return f


def _deep_feed(symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT")):
    return _mk_feed(symbols, settings.CANDLE_LIMIT)


def _shallow_feed(symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT")):
    """The production deadlock state: 1-bar keys created by WS events."""
    return _mk_feed(symbols, 1)


def _mid_feed(symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT")):
    """100 bars: passes the old 60-bar fresh() gate, serves NOTHING at
    get_cached(limit=CANDLE_LIMIT) - the dishonest-metric case."""
    return _mk_feed(symbols, 100)


def _fake_candles(n):
    idx = pd.date_range("2026-01-01", periods=n, freq="1h",
                        tz="UTC", name="open_time")
    return pd.DataFrame({
        "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
        "volume": 1.0, "close_time": idx + pd.Timedelta(hours=1),
        "quote_volume": 1.0, "trades": 1,
        "taker_buy_base": 0.5, "taker_buy_quote": 0.5,
    }, index=idx)


@pytest.fixture()
def gate_env(monkeypatch):
    """Isolated gate: fake rate_limiter + fake ws_feed module attribute."""
    rl = importlib.import_module("src.core.rate_limiter")
    fake_ws = SimpleNamespace(
        is_live=lambda: False,
        servable_coverage=lambda min_bars=None: 0.0,
        cache_status=lambda: {"pending_keys": 0},
        ensure_seed_worker=lambda: False,
        _universe=[],
    )
    monkeypatch.setattr(rl, "rate_limiter",
                        SimpleNamespace(cooldown_remaining=lambda: 0.0))
    monkeypatch.setattr(wsmod, "ws_feed", fake_ws)
    cycmod.reset_gate_state()
    env = {"ws": fake_ws, "rl": rl}
    return env


def _cooldown(env, seconds):
    env["rl"].rate_limiter = SimpleNamespace(
        cooldown_remaining=lambda: seconds)


# ---------- 1. depth-aware seeder ----------

class TestDepthAwareMissing:
    def test_shallow_ws_created_keys_are_seeder_work(self):
        """THE deadlock: 1-bar WS keys must count as missing."""
        f = _shallow_feed()
        assert len(f._missing_keys()) == 3 * 2

    def test_deep_keys_are_not_missing(self):
        assert _deep_feed()._missing_keys() == []

    def test_failed_key_excluded_then_retried_after_window(self):
        f = _shallow_feed()
        f._seed_failed[f._key("BTCUSDT", "1h")] = time.monotonic()
        assert ("BTCUSDT", "1h") not in f._missing_keys()
        f._seed_failed[f._key("BTCUSDT", "1h")] = time.monotonic() - (
            max(60.0, float(settings.WS_SEED_RETRY_AFTER_S)) + 1)
        assert ("BTCUSDT", "1h") in f._missing_keys()

    def test_short_history_key_excluded_until_universe_change(self, monkeypatch):
        f = _shallow_feed()
        f._seed_short.add(f._key("BTCUSDT", "1h"))
        assert ("BTCUSDT", "1h") not in f._missing_keys()
        # v5.24's lazy trigger would start a REAL worker thread here (the
        # feed has missing keys) - mock it so no thread leaks into other
        # tests and pokes the DataFetcher patch below.
        class _NoT:
            def __init__(self, *a, **k):
                pass
            def start(self):
                pass
        monkeypatch.setattr(wsmod.threading, "Thread", _NoT)
        f.update_universe(["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"])
        assert f._key("BTCUSDT", "1h") not in f._seed_short

    def test_seeder_records_failure_and_short_history(self, monkeypatch):
        f = WSKlineFeed()
        f._started = True
        f._universe = ["BADUSDT", "NEWUSDT"]
        calls = {"n": 0}

        def fake_rest(sym, interval, limit):
            if sym not in ("BADUSDT", "NEWUSDT"):
                # a leaked background worker from another test hitting the
                # class-level patch - serve it harmlessly, never count it
                return _fake_candles(settings.CANDLE_LIMIT)
            calls["n"] += 1
            if sym == "BADUSDT":
                raise RuntimeError("delisted")
            return _fake_candles(40)          # brand-new listing: 40 bars

        from src.core.data_fetcher import DataFetcher
        monkeypatch.setattr(DataFetcher, "_get_candles_rest",
                            staticmethod(fake_rest))
        monkeypatch.setattr(settings, "WS_SEED_DELAY_S", 0.0)
        rl = importlib.import_module("src.core.rate_limiter")

        class _RL:
            @staticmethod
            def cooldown_remaining():
                return 0.0
        monkeypatch.setattr(rl, "rate_limiter", _RL())

        f._seed_missing()                      # runs in-line

        assert calls["n"] == 4                 # 1 attempt per KEY (2x2), no spin
        assert f._seed_failed.get(f._key("BADUSDT", "1h")) is not None
        assert f._seed_failed.get(f._key("BADUSDT", "4h")) is not None
        # NEWUSDT ingested its 40 bars and was marked short-history
        assert len(f._bars[f._key("NEWUSDT", "1h")]) == 40
        assert f._key("NEWUSDT", "1h") in f._seed_short
        # sweep is DONE: no key remains pending
        assert f._missing_keys() == []

    def test_worker_restarts_without_reconnect(self, monkeypatch):
        """v5.24: update_universe triggers the seeder even when the merged
        set is unchanged and no reconnect fires (the old code only started
        the worker from _on_open)."""
        f = _deep_feed()
        f._bars.pop(f._key("ETHUSDT", "1h"))   # one missing key
        started = {"flag": False}

        class _T:
            def __init__(self, *a, **k):
                pass
            def start(self):
                started["flag"] = True
        monkeypatch.setattr(wsmod.threading, "Thread", _T)
        f.update_universe(["BTCUSDT", "ETHUSDT", "SOLUSDT"])  # unchanged set
        assert started["flag"] is True


# ---------- 2. honest servable metric ----------

class TestServableCoverage:
    def test_shallow_cache_serves_nothing(self):
        f = _shallow_feed()
        assert f.coverage() == 0.0
        assert f.servable_coverage() == 0.0

    def test_mid_depth_cache_exposes_the_metric_lie(self):
        """100 bars: the OLD metric said 100% ready, the analyzer would
        still have served zero symbols (get_cached wants CANDLE_LIMIT)."""
        f = _mid_feed()
        assert f.coverage() == 1.0
        assert f.servable_coverage() == 0.0

    def test_deep_fresh_cache_fully_servable(self):
        assert _deep_feed().servable_coverage() == 1.0

    def test_stale_deep_cache_not_servable(self):
        f = _deep_feed()
        f._last_event[f._key("BTCUSDT", "4h")] = time.monotonic() - 9999
        assert f.servable_coverage() == pytest.approx(2 / 3)

    def test_one_shallow_interval_disqualifies_symbol(self):
        f = _deep_feed(symbols=("BTCUSDT",))
        f._bars[f._key("BTCUSDT", "4h")] = [[0] * 12]
        assert f.servable_coverage() == 0.0

    def test_cache_status_shape(self):
        st = _shallow_feed().cache_status()
        assert st["pending_keys"] == 6
        assert st["missing_keys"] == 0 and st["shallow_keys"] == 6
        assert st["servable_symbols"] == 0
        assert st["seeding"] is False

    def test_status_exposes_transparency_fields(self):
        st = _mid_feed().status()
        assert st["servable_pct"] == 0.0
        assert st["cache_status"]["shallow_keys"] == 6


# ---------- 3. lazy worker trigger ----------

class TestEnsureSeedWorker:
    def test_starts_worker_when_missing(self, monkeypatch):
        f = _shallow_feed()
        started = {"flag": False}

        class _T:
            def __init__(self, *a, **k):
                pass
            def start(self):
                started["flag"] = True
        monkeypatch.setattr(wsmod.threading, "Thread", _T)
        assert f.ensure_seed_worker() is True
        assert started["flag"] is True

    def test_noop_when_cache_complete(self):
        assert _deep_feed().ensure_seed_worker() is False

    def test_noop_when_not_started(self):
        assert WSKlineFeed().ensure_seed_worker() is False

    def test_noop_when_worker_already_busy(self):
        f = _shallow_feed()
        f._seed_busy = True
        assert f.ensure_seed_worker() is False


# ---------- 4. the gate ----------

class TestGateAntiDeadlock:
    def test_rest_available_cache_complete_runs(self, gate_env):
        ws = gate_env["ws"]
        ws.is_live = lambda: True
        ws.servable_coverage = lambda min_bars=None: 1.0
        assert cycmod.rate_limit_gate() == "run"
        assert cycmod.gate_skip_streak() == 0

    def test_recovery_window_yielded_to_seeder(self, gate_env):
        """REST just lifted, cache shallow -> yield, don't burn the window."""
        ws = gate_env["ws"]
        ws.is_live = lambda: True
        ws.servable_coverage = lambda min_bars=None: 0.0
        ws.cache_status = lambda: {"pending_keys": 86}
        fired = {"seeder": False}
        ws.ensure_seed_worker = lambda: fired.update(seeder=True) or True
        assert cycmod.rate_limit_gate() == "skip"
        assert cycmod.gate_skip_streak() == 1
        assert fired["seeder"] is True

    def test_yield_stall_breaker_falls_through_to_run(self, gate_env):
        """After GATE_RECOVERY_SKIP_LIMIT yields, normal 'run' resumes."""
        ws = gate_env["ws"]
        ws.is_live = lambda: True
        ws.servable_coverage = lambda min_bars=None: 0.0
        ws.cache_status = lambda: {"pending_keys": 86}
        for _ in range(settings.GATE_RECOVERY_SKIP_LIMIT):
            assert cycmod.rate_limit_gate() == "skip"
        assert cycmod.rate_limit_gate() == "run"
        assert cycmod.gate_skip_streak() == 0

    def test_no_yield_without_pending_keys(self, gate_env):
        ws = gate_env["ws"]
        ws.is_live = lambda: True
        ws.servable_coverage = lambda min_bars=None: 0.0
        ws.cache_status = lambda: {"pending_keys": 0}
        assert cycmod.rate_limit_gate() == "run"

    def test_no_yield_when_ws_down(self, gate_env):
        """WS down + REST available -> run (REST is the only option)."""
        ws = gate_env["ws"]
        ws.is_live = lambda: False
        assert cycmod.rate_limit_gate() == "run"

    def test_cooldown_full_servable_degraded(self, gate_env):
        ws = gate_env["ws"]
        ws.is_live = lambda: True
        ws.servable_coverage = lambda min_bars=None: 1.0
        _cooldown(gate_env, 600.0)
        assert cycmod.rate_limit_gate() == "degraded"
        assert cycmod.gate_skip_streak() == 0

    def test_cooldown_shallow_cache_skips_then_escape_hatch(self, gate_env):
        """THE production deadlock: cooldown + shallow cache. After
        GATE_ESCAPE_SKIP_MIN skips, a partial WS-only cycle runs."""
        ws = gate_env["ws"]
        ws.is_live = lambda: True
        ws._universe = ["S%d" % i for i in range(43)]
        ws.servable_coverage = lambda min_bars=None: 0.05   # ~2 symbols
        ws.cache_status = lambda: {"pending_keys": 84}
        _cooldown(gate_env, 600.0)
        outcomes = [cycmod.rate_limit_gate() for _ in range(4)]
        assert outcomes[:3] == ["skip", "skip", "skip"]
        assert outcomes[3] == "degraded"       # the escape hatch

    def test_escape_hatch_latches_for_cooldown_duration(self, gate_env):
        """Once latched, every subsequent tick during the SAME cooldown
        keeps analyzing what exists (no skip/degraded/skip/degraded flap)."""
        ws = gate_env["ws"]
        ws.is_live = lambda: True
        ws._universe = ["S%d" % i for i in range(43)]
        ws.servable_coverage = lambda min_bars=None: 0.05
        ws.cache_status = lambda: {"pending_keys": 84}
        _cooldown(gate_env, 600.0)
        for _ in range(3):
            cycmod.rate_limit_gate()
        assert cycmod.rate_limit_gate() == "degraded"
        assert cycmod.rate_limit_gate() == "degraded"      # stays latched

    def test_escape_hatch_needs_one_servable(self, gate_env):
        ws = gate_env["ws"]
        ws.is_live = lambda: True
        ws.servable_coverage = lambda min_bars=None: 0.0
        ws.cache_status = lambda: {"pending_keys": 86}
        _cooldown(gate_env, 600.0)
        outcomes = [cycmod.rate_limit_gate() for _ in range(5)]
        assert outcomes == ["skip"] * 5        # nothing servable -> still skip

    def test_run_resets_streak(self, gate_env):
        ws = gate_env["ws"]
        ws.is_live = lambda: True
        ws.servable_coverage = lambda min_bars=None: 0.0
        ws.cache_status = lambda: {"pending_keys": 86}
        assert cycmod.rate_limit_gate() == "skip"          # yield #1
        ws.servable_coverage = lambda min_bars=None: 1.0
        assert cycmod.rate_limit_gate() == "run"           # resets
        ws.servable_coverage = lambda min_bars=None: 0.0
        assert cycmod.gate_skip_streak() == 0

    def test_ws_down_cooldown_still_skips(self, gate_env):
        ws = gate_env["ws"]
        ws.is_live = lambda: False
        _cooldown(gate_env, 300.0)
        assert cycmod.rate_limit_gate() == "skip"


# ---------- 5. settings + dashboard contract ----------

class TestContract:
    def test_v524_knobs_exist(self):
        assert settings.WS_SEED_RETRY_AFTER_S == 1800
        assert settings.GATE_RECOVERY_SKIP_LIMIT == 4
        assert settings.GATE_ESCAPE_SKIP_MIN == 3
        assert settings.GATE_MIN_SERVABLE == 1

    def test_dashboard_transparency_fields_exist(self):
        src = Path("src/web/app.py").read_text(encoding="utf-8")
        assert "gate_skip_streak" in src
        wsrc = Path("src/core/ws_feed.py").read_text(encoding="utf-8")
        assert "servable_pct" in wsrc and "cache_status" in wsrc
