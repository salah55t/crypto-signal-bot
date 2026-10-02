"""
v5.31 WS-tracked tests.

Production 2026-10-03: the ban persisted on the shared Render egress IP and
the user directive landed: "reduce the number of requests to avoid the ban,
without touching strategy quality, and let tracking come from WebSocket."

The recurring REST audit found (steady state, no ban):
  - scalp: 15 shortlist symbols x weight-5 1000-bar 1s-klines EVERY tick
    (<= 75 w/min, the single largest recurring REST cost)
  - momentum: DOUBLE_IND_TIMEFRAME=1m klines for the universe per cycle
    (1m was never WS-covered)
  - MTF: "1d" klines (250 bars) per candidate per cycle (never WS-covered)

v5.31 moves ALL of them onto the WS feed (same Binance candles, zero signal
logic change): WS_INTERVALS now covers every recurring consumer, the seeder
targets each interval's true depth, the scalp reads 1s bars from the WS
cache with a zero-REST warmup rule, and a per-path REST spend ledger makes
the remaining spend visible on /api/health.

Covered here:
  - settings: WS_INTERVALS covers every consumer
  - ws_feed: per-interval deque cap + depth target (1s -> 1000), seeder
    and missing-keys depth semantics, public is_active/subscribed helpers
  - scalp: zero-REST while the WS cache warms; WS cache served without
    REST; REST fallback only when the feed is down
  - binance_client: per-path spend ledger + window aggregation
  - data_fetcher: WS-covered intervals (1d) never hit REST when warm
  - /api/health: rest_spend_1h exposed

NOTE (mandatory project pattern): singletons are shared across the suite -
monkeypatch private state on them and let monkeypatch undo it.
"""
import importlib
import time
from collections import deque
from contextlib import contextmanager

import pandas as pd
import pytest


def _mod(name: str):
    return importlib.import_module(name)


@contextmanager
def _patch_static(cls, name, fn):
    """Patch a staticmethod class attr and restore the EXACT descriptor.

    monkeypatch.setattr(cls, name, staticmethod(fn)) round-trips through
    getattr()/setattr(), so the undo writes back the PLAIN FUNCTION instead
    of the staticmethod descriptor - instance-level callers then bind self
    and every later call arg-shifts. Saving the raw __dict__ entry and
    restoring it verbatim is the only exact undo.
    """
    orig = cls.__dict__[name]
    setattr(cls, name, staticmethod(fn))
    try:
        yield
    finally:
        setattr(cls, name, orig)


# ----------------------------------------------------------------------
# Settings: the WS interval set covers every recurring consumer
# ----------------------------------------------------------------------
def test_ws_intervals_cover_every_consumer():
    from config.settings import settings
    ivs = set(settings.WS_INTERVALS)
    assert settings.TIMEFRAMES[0] in ivs          # 4h strategies
    assert "1h" in ivs                            # map / router / filter
    assert settings.DOUBLE_IND_TIMEFRAME in ivs   # momentum scanner
    assert "1d" in ivs                            # MTF daily confluence
    assert "1s" in ivs                            # micro-scalp (armed)
    # sorted for deterministic stream URLs
    assert settings.WS_INTERVALS == sorted(settings.WS_INTERVALS)


# ----------------------------------------------------------------------
# ws_feed: per-interval depth + cap semantics
# ----------------------------------------------------------------------
@pytest.fixture
def feed(monkeypatch, tmp_path):
    mod = _mod("src.core.ws_feed")
    f = mod.WSKlineFeed()
    f._started = True
    return mod, f


def test_depth_target_and_maxlen_per_interval(feed):
    mod, f = feed
    from config.settings import settings
    assert f._depth_target("1s") == int(settings.SCALP_KLINES_LIMIT)
    assert f._depth_target("4h") == int(settings.CANDLE_LIMIT)
    assert f._depth_target("1m") == int(settings.CANDLE_LIMIT)
    assert f._maxlen_for("1s") == mod.WSKlineFeed.MAX_BARS_1S
    assert f._maxlen_for("1h") == mod.WSKlineFeed.MAX_BARS
    # the 1s cap must hold the FULL scalp read or get_cached never serves
    assert mod.WSKlineFeed.MAX_BARS_1S >= int(settings.SCALP_KLINES_LIMIT)


def test_missing_keys_use_per_interval_depth(feed):
    mod, f = feed
    f._universe = ["AAAUSDT"]
    # 300 bars satisfy CANDLE_LIMIT intervals but NOT the 1s depth target
    for iv in ["1h", "4h", "1m", "1d"]:
        f._bars[f._key("AAAUSDT", iv)] = deque(
            [[0] * 12 for _ in range(300)], maxlen=f._maxlen_for(iv))
        f._last_event[f._key("AAAUSDT", iv)] = time.monotonic()
    f._bars[f._key("AAAUSDT", "1s")] = deque(
        [[0] * 12 for _ in range(300)], maxlen=f._maxlen_for("1s"))
    f._last_event[f._key("AAAUSDT", "1s")] = time.monotonic()
    missing = dict.fromkeys(f._missing_keys())
    assert ("AAAUSDT", "1h") not in missing
    assert ("AAAUSDT", "1s") in missing       # 300 < 1000 - still seeder work
    # at full scalp depth the 1s key stops being missing
    f._bars[f._key("AAAUSDT", "1s")] = deque(
        [[0] * 12 for _ in range(1000)], maxlen=f._maxlen_for("1s"))
    assert ("AAAUSDT", "1s") not in dict.fromkeys(f._missing_keys())


def test_seed_fetches_scalp_depth_for_1s(feed, monkeypatch):
    mod, f = feed
    from src.core.data_fetcher import DataFetcher
    from config.settings import settings
    f._universe = ["AAAUSDT"]
    calls = []

    def fake_rest(sym, interval, limit=200):
        calls.append((sym, interval, limit))
        return pd.DataFrame({"x": range(limit)})

    captured = []

    def fake_ingest(sym, df, interval="1h"):
        captured.append((sym, interval, len(df)))
        with f._lock:
            f._bars[f._key(sym, interval)] = deque(
                [[0] * 12 for _ in range(len(df))],
                maxlen=f._maxlen_for(interval))
            f._last_event[f._key(sym, interval)] = time.monotonic()

    monkeypatch.setattr(DataFetcher, "_get_candles_rest",
                        staticmethod(fake_rest))
    monkeypatch.setattr(f, "ingest", fake_ingest)
    monkeypatch.setattr(settings, "WS_SEED_DELAY_S", 0.01)
    f._seed_missing()
    by_iv = {iv: lim for _, iv, lim in calls}
    assert by_iv["1s"] == int(settings.SCALP_KLINES_LIMIT)
    assert by_iv["1h"] == int(settings.CANDLE_LIMIT)
    assert f._key("AAAUSDT", "1s") not in f._seed_short
    assert f._missing_keys() == []


def test_seed_marks_truly_short_1s_history(feed, monkeypatch):
    mod, f = feed
    from src.core.data_fetcher import DataFetcher
    from config.settings import settings
    f._universe = ["NEWUSDT"]     # a fresh listing: little 1s history

    def fake_rest(sym, interval, limit=200):
        return pd.DataFrame({"x": range(120)})   # history exhausted

    monkeypatch.setattr(DataFetcher, "_get_candles_rest",
                        staticmethod(fake_rest))

    def fake_ingest(sym, df, interval="1h"):
        with f._lock:
            f._bars[f._key(sym, interval)] = deque(
                [[0] * 12 for _ in range(len(df))],
                maxlen=f._maxlen_for(interval))
            f._last_event[f._key(sym, interval)] = time.monotonic()

    monkeypatch.setattr(f, "ingest", fake_ingest)
    monkeypatch.setattr(settings, "WS_SEED_DELAY_S", 0.01)
    f._seed_missing()
    # every interval is short vs its target -> marked, no re-burn
    assert f._seed_short
    assert f._missing_keys() == []


def test_reseed_refetches_scalp_depth_for_1s(feed, monkeypatch):
    mod, f = feed
    from src.core.data_fetcher import DataFetcher
    f._universe = ["AAAUSDT"]
    with f._lock:
        f._bars[f._key("AAAUSDT", "1s")] = deque([[0] * 12], maxlen=1100)
    f._needs_reseed = True
    calls = []

    def fake_rest(sym, interval, limit=200):
        calls.append((sym, interval, limit))
        return pd.DataFrame({"x": range(limit)})

    monkeypatch.setattr(DataFetcher, "_get_candles_rest",
                        staticmethod(fake_rest))
    monkeypatch.setattr(f, "ingest", lambda *a, **k: None)
    f._reseed_all()
    assert calls and calls[0][2] == mod.WSKlineFeed._depth_target("1s")
    assert f._needs_reseed is False


def test_public_helpers_is_active_and_subscribed(feed):
    mod, f = feed
    f._universe = ["BTCUSDT"]
    f._stop = False
    assert f.is_active() is True
    assert f.subscribed("BTCUSDT", "1h") is True
    assert f.subscribed("BTCUSDT", "1s") is True
    assert f.subscribed("ETHUSDT", "1s") is False   # not in the universe
    assert f.subscribed("BTCUSDT", "15m") is False  # not a subscribed TF
    f._stop = True
    assert f.is_active() is False


# ----------------------------------------------------------------------
# Scalp: zero-REST warmup + WS-first
# ----------------------------------------------------------------------
def _s_rows(n=1000):
    """n one-second rows rising gently (bullish for the scalp strategy)."""
    base_ms = 1_700_000_000_000
    return [[base_ms + i * 1000,
             str(100.0 + i * 0.001), str(100.2 + i * 0.001),
             str(99.8 + i * 0.001), str(100.1 + i * 0.001),
             "10", base_ms + (i + 1) * 1000 - 1,
             "1000", "5", "5", "500", "0"] for i in range(n)]


@pytest.fixture
def scalp(monkeypatch):
    from config.settings import settings
    mod = _mod("src.analysis.scalp_scanner")
    return mod.ScalpScanner(), settings


def test_scalp_zero_rest_while_ws_cache_warms(scalp, monkeypatch):
    scanner, settings = scalp
    ws_mod = _mod("src.core.ws_feed")
    df_mod = _mod("src.core.data_fetcher")

    class _WarmingWS:
        def is_active(self):
            return True

        def subscribed(self, symbol, interval):
            return True

        def get_cached(self, symbol, limit, interval="1h"):
            return None          # seeder still fetching the 1000-bar history

    monkeypatch.setattr(ws_mod, "ws_feed", _WarmingWS())

    def _must_not_rest(*a, **k):
        raise AssertionError("REST fired while the WS cache was warming")

    with _patch_static(df_mod.DataFetcher, "get_candles", _must_not_rest):
        out = scanner._scan_one("BTCUSDT")
    assert out["skip"] is True
    assert "warming" in out["reason"]


def test_scalp_serves_ws_cache_without_rest(scalp, monkeypatch):
    scanner, settings = scalp
    ws_mod = _mod("src.core.ws_feed")
    df_mod = _mod("src.core.data_fetcher")
    from src.core.data_fetcher import DataFetcher

    df = DataFetcher.klines_to_df(_s_rows(int(settings.SCALP_KLINES_LIMIT)))

    class _WarmWS:
        def is_active(self):
            return True

        def subscribed(self, symbol, interval):
            return True

        def get_cached(self, symbol, limit, interval="1h"):
            assert interval == "1s"
            assert limit == int(settings.SCALP_KLINES_LIMIT)
            return df

    monkeypatch.setattr(ws_mod, "ws_feed", _WarmWS())

    def _must_not_rest(*a, **k):
        raise AssertionError("REST fired despite a warm WS cache")

    with _patch_static(df_mod.DataFetcher, "get_candles", _must_not_rest):
        out = scanner._scan_one("BTCUSDT")
    # the strategy ran on WS data (bullish series -> a candidate dict or a
    # strategy-level skip; either way PAST the data step and zero REST)
    assert out is not None


def test_scalp_rest_fallback_only_when_feed_down(scalp, monkeypatch):
    scanner, settings = scalp
    ws_mod = _mod("src.core.ws_feed")
    df_mod = _mod("src.core.data_fetcher")
    from src.core.data_fetcher import DataFetcher

    class _DownWS:
        def is_active(self):
            return False           # feed disabled/dead

        def subscribed(self, symbol, interval):
            return False

    monkeypatch.setattr(ws_mod, "ws_feed", _DownWS())
    calls = []

    def fake_get_candles(symbol, interval, limit=200):
        calls.append((symbol, interval, limit))
        return DataFetcher.klines_to_df(_s_rows(200))

    with _patch_static(df_mod.DataFetcher, "get_candles", fake_get_candles):
        out = scanner._scan_one("BTCUSDT")
    assert calls, "feed down - the REST fallback is the only data path"
    assert out is not None


# ----------------------------------------------------------------------
# binance_client: per-path REST spend ledger
# ----------------------------------------------------------------------
def test_rest_spend_ledger_window(monkeypatch):
    bc_mod = _mod("src.core.binance_client")
    cl = bc_mod.binance_client
    now = time.monotonic()
    events = deque([
        (now - 10.0, "/api/v3/klines", 5.0),
        (now - 20.0, "/api/v3/klines", 5.0),
        (now - 30.0, "/api/v3/ticker/price", 2.0),
        (now - 3700.0, "/api/v3/time", 1.0),   # outside the 1h window
    ])
    monkeypatch.setattr(cl, "_spend", events)
    out = cl.rest_spend(3600.0)
    assert out["total_calls"] == 3
    assert out["total_weight"] == 12
    assert out["paths"]["/api/v3/klines"] == {"calls": 2, "weight": 10}
    assert "/api/v3/time" not in out["paths"]


def test_rest_spend_recorded_on_real_get(monkeypatch):
    bc_mod = _mod("src.core.binance_client")
    cl = bc_mod.binance_client

    class _Resp:
        status_code = 200
        text = "ok"
        headers = {}
        def raise_for_status(self):
            return None
        def json(self):
            return {}

    monkeypatch.setattr(cl.session, "get",
                        lambda url, **k: _Resp(), raising=True)
    before = cl.request_count
    cl._get("/api/v3/ping")
    assert cl.request_count == before + 1
    spend = cl.rest_spend(3600.0)
    assert spend["total_calls"] >= 1
    assert "/api/v3/ping" in spend["paths"]


# ----------------------------------------------------------------------
# data_fetcher: WS-covered consumers never hit REST when warm
# ----------------------------------------------------------------------
def test_mtf_daily_reads_ws_cache_zero_rest(monkeypatch):
    df_mod = _mod("src.core.data_fetcher")
    ws_mod = _mod("src.core.ws_feed")
    from src.core.data_fetcher import DataFetcher

    class _WS:
        def get_cached(self, symbol, limit, interval="1h"):
            return DataFetcher.klines_to_df(_s_rows(min(limit, 1000)))

    monkeypatch.setattr(ws_mod, "ws_feed", _WS())

    def _must_not_rest(*a, **k):
        raise AssertionError("1d klines hit REST despite the WS cache")

    with _patch_static(DataFetcher, "_get_candles_rest", _must_not_rest):
        df = DataFetcher.get_candles("BTCUSDT", "1d", 250)
    assert df is not None and len(df) == 250


# ----------------------------------------------------------------------
# /api/health: rest_spend_1h exposed
# ----------------------------------------------------------------------
def test_health_exposes_rest_spend():
    from fastapi.testclient import TestClient
    from src.web.app import app
    client = TestClient(app)
    r = client.get("/api/health")
    assert r.status_code == 200
    rl = (r.json().get("rate_limit") or {})
    assert "rest_spend_1h" in rl
    assert "total_weight" in rl["rest_spend_1h"]
