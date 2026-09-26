"""
v5.12 ban-proof tests.

Production incident (2026-09-26): a Binance 429 with Retry-After >= 600s was
capped locally at a 120s cooldown, so the bot poked the still-hot shared IP
every 2 minutes until Binance escalated to a 418 IP auto-ban. During the
resulting 598s ban every queued get_candles call logged its own WARNING
("Rate-limited on get_candles with 598s ban left - aborting fast" x hundreds)
and the analysis burst overwrote data/recommendations.json with an empty
snapshot.

Fixes covered:
  1. binance_client: 429 honors Retry-After fully (no 120s cap)
  2. helpers.retry_on_failure: ban-abort warning throttled to 1 line / 60s
  3. ws_feed.get_cached(allow_stale=True): serves bounded-stale series
  4. data_fetcher.get_candles: stale WS cache served during a ban; REST
     path fast-fails without touching the network while banned
  5. analyzer.analyze_all: burst aborts mid-ban, last good snapshot kept
     (recommendations.json NOT overwritten), last_run_aborted flag set
  6. bottom_scanner.scan: clean empty result during a ban
"""
import importlib
import json
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.rate_limiter import WeightedRateLimiter, RateLimitError

dfmod = importlib.import_module("src.core.data_fetcher")  # the MODULE
wsm = importlib.import_module("src.core.ws_feed")
import src.utils.helpers as helpers
from src.analysis.analyzer import MarketAnalyzer, _ban_active

# ============================================
# helpers
# ============================================


def make_limiter(budget=100.0, reserve=20.0):
    return WeightedRateLimiter(budget_per_min=budget,
                               hard_ceiling=6000,
                               priority_reserve=reserve)


@pytest.fixture
def clean_ban_log():
    helpers._BAN_LOG_LAST_TS = 0.0
    helpers._BAN_LOG_SUPPRESSED = 0
    yield
    helpers._BAN_LOG_LAST_TS = 0.0
    helpers._BAN_LOG_SUPPRESSED = 0


def make_feed_with_bars(symbol="BTCUSDT", interval="1h", n=120,
                        age_s=0.0, started=True):
    """A WSKlineFeed with seeded bars and a controllable last-event age."""
    feed = wsm.WSKlineFeed()
    feed._started = started
    rows = [[1_700_000_000_000 + i * 3_600_000, 1, 2, 0.5, 1.5, 10,
             1_700_003_599_000, 10, 5, 5, 5, "rest"] for i in range(n)]
    key = feed._key(symbol, interval)
    from collections import deque
    with feed._lock:
        feed._bars[key] = deque(rows, maxlen=400)
        feed._last_event[key] = time.monotonic() - age_s
    return feed


# ============================================
# 1. 429 honors Retry-After fully
# ============================================

class _FakeResp:
    def __init__(self, status, retry_after=None, used=None):
        self.status_code = status
        self.headers = {}
        if retry_after is not None:
            self.headers["Retry-After"] = str(retry_after)
        if used is not None:
            self.headers["X-MBX-USED-WEIGHT-1M"] = str(used)
        self.text = "fake"

    def json(self):
        return {}

    def raise_for_status(self):
        return None


def _binance_module():
    """The binance_client MODULE (src.core re-exports the singleton under
    the same name, so plain 'import ... as' can bind the wrong object)."""
    return importlib.import_module("src.core.binance_client")


def test_429_honors_retry_after_fully(monkeypatch):
    """429 + Retry-After=600 must cool down ~600s, not the old 120s cap."""
    bc = _binance_module()
    limiter = make_limiter()
    monkeypatch.setattr(bc, "rate_limiter", limiter, raising=False)
    monkeypatch.setattr(
        bc.binance_client.session, "get",
        lambda *a, **k: _FakeResp(429, retry_after=600))

    with pytest.raises(RateLimitError):
        bc.binance_client._get("/api/v3/ping")

    left = limiter.cooldown_remaining()
    assert 590.0 <= left <= 602.0, f"expected ~600s cooldown, got {left:.0f}s"


def test_429_without_header_gets_sane_default(monkeypatch):
    bc = _binance_module()
    limiter = make_limiter()
    monkeypatch.setattr(bc, "rate_limiter", limiter, raising=False)
    monkeypatch.setattr(
        bc.binance_client.session, "get",
        lambda *a, **k: _FakeResp(429, retry_after=None))

    with pytest.raises(RateLimitError):
        bc.binance_client._get("/api/v3/ping")

    assert 25.0 <= limiter.cooldown_remaining() <= 35.0


def test_418_still_bans_hard(monkeypatch):
    """418 keeps the >= 900s hard backoff."""
    bc = _binance_module()
    limiter = make_limiter()
    monkeypatch.setattr(bc, "rate_limiter", limiter, raising=False)
    monkeypatch.setattr(
        bc.binance_client.session, "get",
        lambda *a, **k: _FakeResp(418, retry_after=60))

    with pytest.raises(RateLimitError):
        bc.binance_client._get("/api/v3/ping")

    assert limiter.cooldown_remaining() >= 890.0


def test_429_cooldown_blocks_even_priority_calls(monkeypatch):
    """During a server ban nothing is sent - priority lane included."""
    bc = _binance_module()
    limiter = make_limiter()
    monkeypatch.setattr(bc, "rate_limiter", limiter, raising=False)
    monkeypatch.setattr(
        bc.binance_client.session, "get",
        lambda *a, **k: _FakeResp(429, retry_after=600))

    with pytest.raises(RateLimitError):
        bc.binance_client._get("/api/v3/ticker/price", priority=True)

    def boom(*a, **k):
        raise AssertionError("network must not be touched during a ban")
    monkeypatch.setattr(bc.binance_client.session, "get", boom)
    with pytest.raises(RateLimitError):
        bc.binance_client._get("/api/v3/ping", priority=True)


# ============================================
# 2. throttled ban-abort logging
# ============================================

def test_ban_abort_log_throttled(clean_ban_log, caplog):
    """N rapid aborts -> exactly 1 WARNING; the rest counted as suppressed."""
    import logging
    with caplog.at_level(logging.WARNING, logger="crypto_bot"):
        for _ in range(10):
            helpers._log_ban_abort("get_candles", 598.0)
    warns = [r for r in caplog.records
             if r.levelno == logging.WARNING and "aborting fast" in r.message]
    assert len(warns) == 1
    assert helpers._BAN_LOG_SUPPRESSED == 9


def test_ban_abort_log_relogs_after_window(clean_ban_log, caplog):
    import logging
    with caplog.at_level(logging.WARNING, logger="crypto_bot"):
        helpers._log_ban_abort("get_candles", 598.0)
        helpers._BAN_LOG_LAST_TS = time.time() - 61.0  # force window expiry
        helpers._log_ban_abort("get_candles", 590.0)
    warns = [r for r in caplog.records
             if r.levelno == logging.WARNING and "aborting fast" in r.message]
    assert len(warns) == 2


# ============================================
# 3. stale WS cache during bans
# ============================================

def test_get_cached_allow_stale_serves_old_series():
    feed = make_feed_with_bars(age_s=3600.0)  # 1h old - way past fresh TTL
    assert feed.get_cached("BTCUSDT", 100, interval="1h") is None  # fresh path
    df = feed.get_cached("BTCUSDT", 100, interval="1h", allow_stale=True,
                         max_age_s=6 * 3600)
    assert df is not None and len(df) == 100


def test_get_cached_allow_stale_respects_max_age():
    feed = make_feed_with_bars(age_s=7 * 3600.0)  # 7h old > 6h cap
    assert feed.get_cached("BTCUSDT", 100, interval="1h",
                           allow_stale=True, max_age_s=6 * 3600) is None


def test_get_candles_serves_stale_cache_during_ban(monkeypatch):
    """During a 598s ban get_candles must return the stale WS series and
    never touch REST."""
    feed = make_feed_with_bars(age_s=1800.0)  # stale but within 6h
    monkeypatch.setattr(dfmod, "ws_feed", feed, raising=False)
    monkeypatch.setattr(wsm, "ws_feed", feed, raising=False)

    limiter = make_limiter()
    limiter.trigger_cooldown(598.0)
    monkeypatch.setattr("src.core.rate_limiter.rate_limiter", limiter,
                        raising=False)

    def boom(*a, **k):
        raise AssertionError("REST must not be called during a ban when a "
                             "stale WS cache exists")
    monkeypatch.setattr(dfmod.binance_client, "get_klines", boom)

    df = dfmod.DataFetcher.get_candles("BTCUSDT", "1h", 100)
    assert df is not None and len(df) == 100


def test_get_candles_rest_fast_fails_during_ban(monkeypatch):
    """The REST path raises immediately while a hard ban is active."""
    limiter = make_limiter()
    limiter.trigger_cooldown(598.0)
    monkeypatch.setattr("src.core.rate_limiter.rate_limiter", limiter,
                        raising=False)

    def boom(*a, **k):
        raise AssertionError("network must not be touched during a ban")
    monkeypatch.setattr(dfmod.binance_client, "get_klines", boom)

    with pytest.raises(RateLimitError):
        dfmod.DataFetcher._get_candles_rest("BTCUSDT", "1h", 100)


# ============================================
# 4. analyzer burst abort keeps the last good snapshot
# ============================================

@pytest.fixture
def isolated_rec_file(monkeypatch, tmp_path):
    rec = tmp_path / "recommendations.json"
    rec.write_text(json.dumps({"top_recommendations": [{"symbol": "OLD"}]}),
                   encoding="utf-8")
    # NOTE: src.analysis re-exports the analyzer singleton under the module
    # name, so string targets resolve to the INSTANCE - patch via importlib.
    anmod = importlib.import_module("src.analysis.analyzer")
    monkeypatch.setattr(anmod, "RECOMMENDATIONS_FILE", rec)
    return rec


def _analyzer_with_symbols(monkeypatch, symbols):
    an = MarketAnalyzer.__new__(MarketAnalyzer)  # skip __init__ (no I/O)
    an.excluded = set()
    an._symbols_ts = None
    an.symbols = list(symbols)
    an.last_run_aborted = False
    return an


def test_analyze_all_aborts_and_keeps_snapshot(monkeypatch, isolated_rec_file):
    """A mid-burst ban aborts the burst, leaves recommendations.json intact
    and raises the last_run_aborted flag."""
    limiter = make_limiter()
    monkeypatch.setattr("src.core.rate_limiter.rate_limiter", limiter,
                        raising=False)
    # ban is already active -> every analyze_one must short-circuit
    limiter.trigger_cooldown(598.0)

    an = _analyzer_with_symbols(monkeypatch, [f"S{i}USDT" for i in range(20)])

    def fake_analyze_one(self, symbol, ws_only=False):
        if _ban_active():
            return {"symbol": symbol, "skip": True, "reason": "rate ban"}
        return {"symbol": symbol, "skip": False, "confidence": 90}

    monkeypatch.setattr(MarketAnalyzer, "analyze_one", fake_analyze_one)

    out = an.analyze_all(parallel=True, max_workers=4)

    assert out == []
    assert an.last_run_aborted is True
    snap = json.loads(isolated_rec_file.read_text(encoding="utf-8"))
    assert snap["top_recommendations"] == [{"symbol": "OLD"}], \
        "the last good snapshot must NOT be overwritten by an aborted run"


def test_analyze_all_normal_run_clears_flag(monkeypatch, isolated_rec_file):
    limiter = make_limiter()
    monkeypatch.setattr("src.core.rate_limiter.rate_limiter", limiter,
                        raising=False)
    an = _analyzer_with_symbols(monkeypatch, ["AAAUSDT", "BBBUSDT"])
    an.last_run_aborted = True  # stale flag from a previous aborted run

    def fake_analyze_one(self, symbol, ws_only=False):
        return {"symbol": symbol, "skip": False, "confidence": 90,
                "expected_rise_pct": 3.0}

    class _StubScorer:
        @staticmethod
        def filter_signals(results, **k):
            return results

    class _StubDB:
        def log_run(self, **k):
            return 1

        def log_recommendation(self, run_id, rec):
            return None

    monkeypatch.setattr(MarketAnalyzer, "analyze_one", fake_analyze_one)
    anmod = importlib.import_module("src.analysis.analyzer")
    monkeypatch.setattr(anmod, "scorer", _StubScorer(), raising=False)
    monkeypatch.setattr(anmod, "db", _StubDB(), raising=False)
    bsmod = importlib.import_module("src.analysis.bottom_scanner")
    monkeypatch.setattr(bsmod.bottom_scanner, "scan",
                        lambda *a, **k: [], raising=False)
    cfgmod = importlib.import_module("config.settings")  # the MODULE
    monkeypatch.setattr(cfgmod.settings, "BOTTOM_BOOST_ENABLED", False,
                        raising=False)

    out = an.analyze_all(parallel=False)
    assert an.last_run_aborted is False
    assert isinstance(out, list)


# ============================================
# 5. bottom scanner bails cleanly during a ban
# ============================================

def test_bottom_scanner_skipped_during_ban(monkeypatch):
    from src.analysis.bottom_scanner import bottom_scanner
    limiter = make_limiter()
    limiter.trigger_cooldown(598.0)
    monkeypatch.setattr("src.core.rate_limiter.rate_limiter", limiter,
                        raising=False)

    def boom(*a, **k):
        raise AssertionError("scanner must not fetch during a ban")
    monkeypatch.setattr(dfmod.DataFetcher, "get_candles", boom)

    assert bottom_scanner.scan(max_candidates=5) == []
