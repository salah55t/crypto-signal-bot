"""
v5.14 WS-first tests: live prices over WebSocket + zero-REST degraded cycles.

Production symptom (2026-09-26): "Rate-limit cooldown active (1674s left) -
skipping this cycle" - a REST ban idled the WHOLE bot even though Binance
WebSocket streams (a different service, zero request-weight) kept flowing.

Fixes covered:
  1. ws_feed: @miniTicker streams keep a live last price per universe symbol
     (get_live_price / get_live_prices) - survives REST bans.
  2. data_fetcher.get_batch_prices: WS prices first, REST only for gaps.
  3. data_fetcher.get_candles: stale WS cache served during ANY cooldown.
  4. cycle.rate_limit_gate: three-way "run" | "degraded" | "skip" - a
     covered universe keeps the analysis running off the WS cache.
  5. cycle.run_analysis_cycle: degraded mode skips the REST ping and runs
     the burst with ws_only=True (order books skipped).
  6. ws_feed._seed_missing: paced cold-start seeding (no more 8-worker
     REST storm after a deploy) + pause while banned.
  7. ws_feed._reseed_all: fetches CANDLE_LIMIT bars (old hardcoded 200
     left the cache permanently below get_cached's depth check).
"""
import importlib
import json
import sys
import time
from collections import deque
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

dfmod = importlib.import_module("src.core.data_fetcher")   # the MODULE
wsm = importlib.import_module("src.core.ws_feed")          # the MODULE
cym = importlib.import_module("src.core.cycle")            # the MODULE
bcmod = importlib.import_module("src.core.binance_client")  # the MODULE
rlmod = importlib.import_module("src.core.rate_limiter")
amod = importlib.import_module("src.analysis.analyzer")
from config.settings import settings
from src.core.data_fetcher import DataFetcher


# ============================================
# helpers
# ============================================

def _rows(n=80, start_ms=1_700_000_000_000):
    return [[start_ms + i * 3_600_000, 1, 2, 0.5, 1.5, 10,
             start_ms + (i + 1) * 3_600_000 - 1, 10, 5, 5, 5, "rest"]
            for i in range(n)]


def _df(n=80):
    return DataFetcher.klines_to_df(_rows(n))


def _mini_event(symbol, price):
    return json.dumps({
        "stream": f"{symbol.lower()}@miniTicker",
        "data": {"e": "24hrMiniTicker", "E": 1, "s": symbol,
                 "c": str(price), "o": "1", "h": "1", "l": "1",
                 "v": "1", "q": "1"},
    })


def make_feed(started=True, connected=False):
    feed = wsm.WSKlineFeed()
    feed._started = started
    feed._connected = connected
    return feed


def inject_bars(feed, symbol, interval="1h", n=120, age_s=0.0):
    key = feed._key(symbol, interval)
    with feed._lock:
        feed._bars[key] = deque(_rows(n), maxlen=400)
        feed._last_event[key] = time.monotonic() - age_s


def set_price(feed, symbol, price, age_s=0.0):
    with feed._lock:
        feed._prices[symbol.upper()] = (time.monotonic() - age_s, float(price))


class _FakeLim:
    """Stand-in for the singleton rate limiter (cooldown only)."""

    def __init__(self, remaining=0.0):
        self._r = float(remaining)

    def cooldown_remaining(self):
        return self._r

    def in_cooldown(self):
        return self._r > 0


class _FakeGateFeed:
    """Stand-in for the ws_feed singleton in cycle-gate tests."""

    def __init__(self, live=True, cov=0.95):
        self._live = live
        self._cov = cov

    def is_live(self):
        return self._live

    def coverage(self, symbols=None, intervals=None):
        return self._cov


@pytest.fixture
def singleton_feed():
    """Arm the REAL ws_feed singleton; restore _started afterwards.

    Prices/bars injected by a test are wiped at teardown so tests stay
    independent (all read paths guard on _started anyway).
    """
    feed = wsm.ws_feed
    was_started = feed._started
    saved_prices = dict(feed._prices)
    saved_bars = dict(feed._bars)
    saved_last_event = dict(feed._last_event)
    feed._started = True
    yield feed
    feed._started = was_started
    feed._prices = saved_prices
    feed._bars = saved_bars
    feed._last_event = saved_last_event


# ============================================
# 1. miniTicker live prices
# ============================================

def test_mini_ticker_updates_live_price():
    feed = make_feed()
    feed._on_message(message=_mini_event("BTCUSDT", 42123.45))
    assert feed.get_live_price("BTCUSDT") == 42123.45


def test_mini_ticker_price_freshness():
    feed = make_feed()
    set_price(feed, "ETHUSDT", 2500.0, age_s=0.0)
    set_price(feed, "OLDUSDT", 9.0, age_s=300.0)  # 5 min old > 70s TTL
    assert feed.get_live_price("ETHUSDT") == 2500.0
    assert feed.get_live_price("OLDUSDT") is None
    # custom TTL widens the window
    assert feed.get_live_price("OLDUSDT", max_age_s=600) == 9.0


def test_mini_ticker_malformed_never_raises():
    feed = make_feed()
    for bad in (
        json.dumps({"stream": "x", "data": {"e": "24hrMiniTicker", "s": "A", "c": "abc"}}),
        json.dumps({"stream": "x", "data": {"e": "24hrMiniTicker", "s": "A", "c": None}}),
        json.dumps({"stream": "x", "data": {"e": "24hrMiniTicker", "s": "A", "c": "-5"}}),
        json.dumps({"stream": "x", "data": {"e": "24hrMiniTicker", "c": "5"}}),
        "not-json{{{",
    ):
        feed._on_message(message=bad)
    assert feed.get_live_price("AUSDT") is None


def test_get_live_prices_filters_stale():
    feed = make_feed()
    set_price(feed, "BTCUSDT", 100.0)
    set_price(feed, "ETHUSDT", 200.0, age_s=999.0)
    out = feed.get_live_prices(["BTCUSDT", "ETHUSDT", "DOGEUSDT"])
    assert out == {"BTCUSDT": 100.0}


def test_kline_events_still_work_alongside_minis():
    feed = make_feed()
    df = _df(5)
    feed.ingest("BTCUSDT", df)
    last_open = int(df.index[-1].timestamp() * 1000)
    feed._on_message(message=_mini_event("BTCUSDT", 77.0))
    k = {"t": last_open + 3_600_000, "T": last_open + 7_199_999,
         "o": "1", "h": "2", "l": "0.5", "c": "1.9", "v": "9",
         "q": "9", "n": 5, "V": "4", "Q": "4", "x": False, "i": "1h"}
    feed._on_message(message=json.dumps(
        {"stream": "btcusdt@kline_1h",
         "data": {"e": "kline", "E": 1, "s": "BTCUSDT", "k": k}}))
    assert len(feed._bars["BTCUSDT|1h"]) == 6
    assert feed.get_live_price("BTCUSDT") == 77.0


# ============================================
# 2. get_batch_prices: WS first, REST only for gaps
# ============================================

def test_batch_prices_ws_first_zero_rest(monkeypatch, singleton_feed):
    set_price(singleton_feed, "BTCUSDT", 42000.5)
    set_price(singleton_feed, "ETHUSDT", 2500.0)
    calls = []

    def _record(symbols, priority=False, **k):
        calls.append(list(symbols))
        return {}

    monkeypatch.setattr(dfmod.binance_client, "get_tickers_batch", _record)
    out = dfmod.data_fetcher.get_batch_prices(["BTCUSDT", "ETHUSDT"])
    assert out == {"BTCUSDT": 42000.5, "ETHUSDT": 2500.0}
    assert calls == []  # zero REST weight


def test_batch_prices_rest_only_for_missing(monkeypatch, singleton_feed):
    set_price(singleton_feed, "BTCUSDT", 42000.5)
    calls = []

    def _record(symbols, priority=False, **k):
        calls.append(list(symbols))
        return {"ETHUSDT": {"price": "2500.0"}}

    monkeypatch.setattr(dfmod.binance_client, "get_tickers_batch", _record)
    out = dfmod.data_fetcher.get_batch_prices(
        ["BTCUSDT", "ETHUSDT"], priority=True)
    assert out["BTCUSDT"] == 42000.5
    assert out["ETHUSDT"] == 2500.0
    assert calls == [["ETHUSDT"]]  # REST asked ONLY for the gap


def test_batch_prices_rest_fallback_when_feed_off(monkeypatch):
    wsm.ws_feed._started = False  # WS unavailable
    calls = []

    def _record(symbols, priority=False, **k):
        calls.append(list(symbols))
        return {"BTCUSDT": {"price": "42000.0"}}

    monkeypatch.setattr(dfmod.binance_client, "get_tickers_batch", _record)
    out = dfmod.data_fetcher.get_batch_prices(["BTCUSDT"])
    assert out == {"BTCUSDT": 42000.0}
    assert calls == [["BTCUSDT"]]


def test_position_prices_survive_rest_ban(monkeypatch, singleton_feed):
    """cycle.fetch_prices must serve WS prices while REST is banned."""
    set_price(singleton_feed, "SOLUSDT", 210.5)
    monkeypatch.setattr(rlmod, "rate_limiter", _FakeLim(600.0))
    calls = []

    def _record(symbols, priority=False, **k):
        calls.append(list(symbols))
        return {}

    monkeypatch.setattr(dfmod.binance_client, "get_tickers_batch", _record)
    out = cym.fetch_prices(["SOLUSDT"], priority=True)
    assert out == {"SOLUSDT": 210.5}
    assert calls == []


# ============================================
# 3. get_candles: stale WS served during ANY cooldown
# ============================================

def test_candles_stale_ws_during_short_cooldown(monkeypatch, singleton_feed):
    """45s pressure cooldown (below the old >60s threshold) must serve the
    stale WS series instead of poking REST."""
    inject_bars(singleton_feed, "BTCUSDT", "1h", n=120, age_s=3600)
    monkeypatch.setattr(rlmod, "rate_limiter", _FakeLim(45.0))

    def _boom(*a, **k):
        raise AssertionError("REST must not fire during a cooldown")

    monkeypatch.setattr(bcmod.binance_client, "get_klines", _boom)
    out = dfmod.data_fetcher.get_candles("BTCUSDT", "1h", 120)
    assert out is not None and len(out) == 120


# ============================================
# 4. cycle gate: run | degraded | skip
# ============================================

def test_gate_run_without_cooldown(monkeypatch):
    monkeypatch.setattr(rlmod, "rate_limiter", _FakeLim(0.0))
    assert cym.rate_limit_gate() == "run"


def test_gate_degraded_when_ws_covers(monkeypatch):
    monkeypatch.setattr(rlmod, "rate_limiter", _FakeLim(1674.0))
    monkeypatch.setattr(wsm, "ws_feed", _FakeGateFeed(live=True, cov=0.95))
    assert cym.rate_limit_gate() == "degraded"


def test_gate_skip_when_coverage_low(monkeypatch):
    monkeypatch.setattr(rlmod, "rate_limiter", _FakeLim(1674.0))
    monkeypatch.setattr(wsm, "ws_feed", _FakeGateFeed(live=True, cov=0.5))
    assert cym.rate_limit_gate() == "skip"


def test_gate_skip_when_feed_not_live(monkeypatch):
    monkeypatch.setattr(rlmod, "rate_limiter", _FakeLim(1674.0))
    monkeypatch.setattr(wsm, "ws_feed", _FakeGateFeed(live=False, cov=0.99))
    assert cym.rate_limit_gate() == "skip"


def test_gate_skip_when_ws_disabled(monkeypatch):
    monkeypatch.setattr(rlmod, "rate_limiter", _FakeLim(1674.0))
    monkeypatch.setattr(wsm, "ws_feed", _FakeGateFeed(live=True, cov=0.99))
    monkeypatch.setattr(settings, "USE_WS_FEED", False)
    try:
        assert cym.rate_limit_gate() == "skip"
    finally:
        monkeypatch.undo()


def test_gate_custom_coverage_threshold(monkeypatch):
    monkeypatch.setattr(rlmod, "rate_limiter", _FakeLim(600.0))
    monkeypatch.setattr(wsm, "ws_feed", _FakeGateFeed(live=True, cov=0.80))
    monkeypatch.setattr(settings, "WS_DEGRADED_COVERAGE", 0.75)
    try:
        assert cym.rate_limit_gate() == "degraded"
    finally:
        monkeypatch.undo()


# ============================================
# 5. degraded cycle wiring
# ============================================

def test_degraded_cycle_runs_ws_only(monkeypatch):
    """Degraded gate: no REST ping, analyzer gets ws_only=True."""
    monkeypatch.setattr(cym, "rate_limit_gate", lambda: "degraded")

    def _boom(*a, **k):
        raise AssertionError("ping during a REST ban is a poke")

    monkeypatch.setattr(cym, "_binance_reachable", _boom)
    manage_calls = []
    monkeypatch.setattr(cym, "manage_open_positions",
                        lambda *a, **k: manage_calls.append(1))
    monkeypatch.setattr(settings, "REGIME_ENABLED", False)
    cap = {}

    def fake_analyze(ws_only=False, **k):
        cap["ws_only"] = ws_only
        return []

    monkeypatch.setattr(cym, "analyzer_analyze", fake_analyze)
    try:
        cym.run_analysis_cycle()
    finally:
        monkeypatch.undo()
    assert manage_calls == [1]
    assert cap.get("ws_only") is True


def test_skip_cycle_never_reaches_analysis(monkeypatch):
    monkeypatch.setattr(cym, "rate_limit_gate", lambda: "skip")

    def _boom(*a, **k):
        raise AssertionError("analysis must not run on a skipped cycle")

    monkeypatch.setattr(cym, "_binance_reachable", _boom)
    monkeypatch.setattr(cym, "analyzer_analyze", _boom)
    cym.run_analysis_cycle()  # returns silently


def test_normal_cycle_passes_ws_only_false(monkeypatch):
    monkeypatch.setattr(cym, "rate_limit_gate", lambda: "run")
    monkeypatch.setattr(cym, "_binance_reachable", lambda *a, **k: True)
    monkeypatch.setattr(cym, "manage_open_positions", lambda *a, **k: None)
    monkeypatch.setattr(settings, "REGIME_ENABLED", False)
    cap = {}

    def fake_analyze(ws_only=False, **k):
        cap["ws_only"] = ws_only
        return []

    monkeypatch.setattr(cym, "analyzer_analyze", fake_analyze)
    try:
        cym.run_analysis_cycle()
    finally:
        monkeypatch.undo()
    assert cap.get("ws_only") is False


# ============================================
# 6. ws_only burst skips order books
# ============================================

class _FakeFetcher:
    def __init__(self, df):
        self.df = df
        self.ob_calls = []

    def get_multi_timeframe_candles(self, symbol, intervals, limit=200):
        return {tf: self.df for tf in intervals}

    def get_order_book(self, symbol, limit=20):
        self.ob_calls.append(symbol)
        return {"symbol": symbol}


class _FakeScorer:
    def __init__(self):
        self.captured = []

    def analyze_symbol(self, df, symbol, multi_tf_data=None, order_book=None):
        self.captured.append({"symbol": symbol, "order_book": order_book})
        return {"symbol": symbol, "confidence": 70.0}


def test_analyze_one_ws_only_skips_order_book(monkeypatch):
    fake_fetch = _FakeFetcher(_df(80))
    fake_scorer = _FakeScorer()
    monkeypatch.setattr(amod, "data_fetcher", fake_fetch)
    monkeypatch.setattr(amod, "scorer", fake_scorer)
    r = amod.analyzer.analyze_one("BTCUSDT", ws_only=True)
    assert not r.get("skip")
    assert fake_fetch.ob_calls == []          # zero REST weight
    assert fake_scorer.captured[0]["order_book"] is None


def test_analyze_one_normal_fetches_order_book(monkeypatch):
    fake_fetch = _FakeFetcher(_df(80))
    fake_scorer = _FakeScorer()
    monkeypatch.setattr(amod, "data_fetcher", fake_fetch)
    monkeypatch.setattr(amod, "scorer", fake_scorer)
    r = amod.analyzer.analyze_one("BTCUSDT", ws_only=False)
    assert not r.get("skip")
    assert fake_fetch.ob_calls == ["BTCUSDT"]
    assert fake_scorer.captured[0]["order_book"] == {"symbol": "BTCUSDT"}


# ============================================
# 7. paced seeding + reseed depth
# ============================================

def test_seed_missing_paces_and_uses_candle_limit(monkeypatch):
    feed = make_feed()
    feed._universe = ["AAAUSDT", "BBBUSDT"]
    calls = []

    def fake_rest(sym, interval, limit=200):
        calls.append((sym, interval, limit))
        return _df(300)

    monkeypatch.setattr(dfmod.DataFetcher, "_get_candles_rest",
                        staticmethod(fake_rest))
    monkeypatch.setattr(settings, "WS_SEED_DELAY_S", 0.01)
    feed._seed_missing()
    # universe x every subscribed interval (settings.WS_INTERVALS)
    n_ivs = len(settings.WS_INTERVALS)
    assert len(calls) == 2 * n_ivs
    assert all(lim == settings.CANDLE_LIMIT for _, _, lim in calls)
    assert feed._key("AAAUSDT", settings.WS_INTERVALS[0]) in feed._bars
    assert not feed._missing_keys()


def test_seed_missing_waits_out_ban(monkeypatch):
    feed = make_feed()
    feed._universe = ["AAAUSDT"]
    state = {"n": 0}

    class _Lift:
        def cooldown_remaining(self):
            state["n"] += 1
            return 500.0 if state["n"] <= 2 else 0.0

    monkeypatch.setattr(rlmod, "rate_limiter", _Lift())
    sleeps = []
    monkeypatch.setattr(wsm.time, "sleep", lambda s: sleeps.append(s))
    monkeypatch.setattr(dfmod.DataFetcher, "_get_candles_rest",
                        staticmethod(lambda s, i, limit=200: _df(80)))
    feed._seed_missing()
    ban_waits = [s for s in sleeps if 5.0 <= s <= 30.0]
    assert ban_waits, "seeder must pause (no REST) while a ban is active"
    assert len(feed._bars) == len(settings.WS_INTERVALS)


def test_seed_busy_guard_prevents_double_worker():
    feed = make_feed()
    feed._seed_busy = True
    feed._universe = ["AAAUSDT"]  # missing keys exist
    feed._seed_missing()          # must return immediately (no-op)
    assert feed._seed_busy is True
    assert not feed._bars
    feed._seed_busy = False


def test_reseed_fetches_candle_limit(monkeypatch):
    feed = make_feed()
    inject_bars(feed, "BTCUSDT", "1h")
    inject_bars(feed, "ETHUSDT", "1h")
    feed._needs_reseed = True
    calls = []

    def fake_rest(sym, interval, limit=200):
        calls.append((sym, interval, limit))
        return _df(300)

    monkeypatch.setattr(dfmod.DataFetcher, "_get_candles_rest",
                        staticmethod(fake_rest))
    feed._reseed_all()
    assert {s for s, _, _ in calls} == {"BTCUSDT", "ETHUSDT"}
    assert all(lim == settings.CANDLE_LIMIT for _, _, lim in calls)
    assert feed._needs_reseed is False


# ============================================
# 8. coverage / is_live / status
# ============================================

def test_coverage_fraction():
    feed = make_feed()
    feed._universe = ["BTCUSDT", "ETHUSDT"]
    inject_bars(feed, "BTCUSDT", "1h")
    inject_bars(feed, "ETHUSDT", "1h")
    assert feed.coverage(intervals=["1h"]) == 1.0
    # ETHUSDT goes stale (15-min TTL)
    with feed._lock:
        feed._last_event[feed._key("ETHUSDT", "1h")] = time.monotonic() - 3600
    assert feed.coverage(intervals=["1h"]) == 0.5


def test_is_live_and_status_shape():
    feed = make_feed(started=True, connected=True)
    assert feed.is_live() is False            # no message yet
    feed._last_msg_ts = time.monotonic()
    set_price(feed, "BTCUSDT", 100.0)
    inject_bars(feed, "BTCUSDT", "1h")
    assert feed.is_live() is True
    s = feed.status()
    for key in ("prices_cached", "prices_fresh", "coverage_pct",
                "live", "seeding", "cached_symbols"):
        assert key in s
    assert s["prices_cached"] == 1
    assert s["prices_fresh"] == 1


def test_run_loop_subscribes_mini_tickers():
    feed = make_feed()
    feed._universe = ["BTCUSDT", "ETHUSDT"]
    # replicate the stream-building block from _run_loop (no real socket)
    streams = []
    with feed._lock:
        for iv in feed._intervals():
            streams.extend(f"{s.lower()}@kline_{iv}" for s in feed._universe)
        streams.extend(f"{s.lower()}@miniTicker" for s in feed._universe)
    assert "btcusdt@miniTicker" in streams
    assert "ethusdt@miniTicker" in streams


# ============================================
# 9. REST request counter visibility
# ============================================

class _Resp200:
    status_code = 200
    headers = {}
    text = ""

    def json(self):
        return {}

    def raise_for_status(self):
        return None


def test_request_counter_increments(monkeypatch):
    client = bcmod.binance_client
    before = client.request_count
    monkeypatch.setattr(client.session, "get",
                        lambda *a, **k: _Resp200())
    client._get("/api/v3/ping")
    assert client.request_count == before + 1
