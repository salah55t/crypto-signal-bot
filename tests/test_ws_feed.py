"""v5.3 WebSocket kline feed + API pressure relief tests.

Incident: the Render shared egress IP sits near Binance's 6,000 weight/min
ceiling from neighbor traffic; our REST polling (analyzer + bottom scanner
refetched the same 150 symbols' klines TWICE per cycle, plus a weight-80
symbol refresh every cycle) kept colliding with it.

Fix under test:
  - WSKlineFeed: live 1h candle cache fed by WS (bypasses REST budget),
    REST-seeded, with staleness guards and reconnect reseeding.
  - data_fetcher.get_candles serves the cache first, falls back to REST.
  - analyzer.refresh_symbols caches the dynamic list for SYMBOL_REFRESH_MIN.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json

import pandas as pd
import pytest

from config.settings import settings
from src.core.data_fetcher import DataFetcher, data_fetcher
from src.core.ws_feed import WSKlineFeed, ws_feed


# ---------- helpers ----------

def _synthetic_rows(n=200, start_ms=None, interval_ms=3_600_000, base=100.0):
    """Synthetic raw kline rows (12-field, like Binance REST)."""
    start_ms = start_ms or int(1_700_000_000_000)
    rows = []
    px = base
    for i in range(n):
        o = px
        c = px * 1.01
        rows.append([
            start_ms + i * interval_ms, o, c * 1.02, c * 0.98, c,
            10.0 + i, start_ms + (i + 1) * interval_ms - 1,
            o * 12.0, 100, 5.0, o * 6.0, "rest",
        ])
        px = c
    return rows


def _event(symbol, open_ms, close, prev_open_ms=None):
    """A Binance combined-stream kline message as a JSON string."""
    k = {
        "t": open_ms, "T": open_ms + 3_599_999,
        "o": str(close * 0.99), "h": str(close * 1.01),
        "l": str(close * 0.98), "c": str(close),
        "v": "12.5", "q": str(close * 12.5),
        "n": 42, "V": "6.0", "Q": str(close * 6.0),
        "x": False, "i": "1h",
    }
    return json.dumps({
        "stream": f"{symbol.lower()}@kline_1h",
        "data": {"e": "kline", "E": open_ms, "s": symbol, "k": k},
    })


@pytest.fixture
def feed():
    """A fresh, disabled-until-touched feed instance."""
    return WSKlineFeed()


# ---------- event handling ----------

def test_event_replaces_inprogress_bar(feed):
    df = DataFetcher.klines_to_df(_synthetic_rows(5))
    feed.ingest("BTCUSDT", df)
    last_open = int(df.index[-1].timestamp() * 1000)
    feed._on_message(message=_event("BTCUSDT", last_open, 123.45))
    bars = feed._bars["BTCUSDT|1h"]   # v5.5: per-(symbol, interval) keys
    assert len(bars) == 5                       # replaced, not appended
    assert float(bars[-1][4]) == 123.45         # updated close


def test_event_appends_new_bar(feed):
    df = DataFetcher.klines_to_df(_synthetic_rows(5))
    feed.ingest("BTCUSDT", df)
    last_open = int(df.index[-1].timestamp() * 1000)
    feed._on_message(message=_event("BTCUSDT", last_open + 3_600_000, 130.0))
    assert len(feed._bars["BTCUSDT|1h"]) == 6
    assert float(feed._bars["BTCUSDT|1h"][-1][4]) == 130.0


def test_event_ignores_stale_open_time(feed):
    df = DataFetcher.klines_to_df(_synthetic_rows(5))
    feed.ingest("BTCUSDT", df)
    before = list(feed._bars["BTCUSDT|1h"])
    old_open = int(df.index[0].timestamp() * 1000)
    feed._on_message(message=_event("BTCUSDT", old_open, 99.0))
    assert list(feed._bars["BTCUSDT|1h"]) == before  # duplicate/stale -> untouched


def test_malformed_message_never_raises(feed):
    feed._on_message(message="not-json{{{")
    feed._on_message(message=json.dumps({"stream": "x", "data": {"e": "trade"}}))
    assert not feed._bars


# ---------- get_cached guards ----------

def test_get_cached_returns_full_series(feed):
    df = DataFetcher.klines_to_df(_synthetic_rows(200))
    feed.ingest("ETHUSDT", df)
    feed._started = True                    # fresh() requires a started feed
    out = feed.get_cached("ETHUSDT", 200)
    assert out is not None
    assert len(out) == 200
    assert list(out.columns) == [
        "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_buy_base", "taker_buy_quote",
    ]


def test_get_cached_none_when_short(feed):
    df = DataFetcher.klines_to_df(_synthetic_rows(50))
    feed.ingest("ETHUSDT", df)
    assert feed.get_cached("ETHUSDT", 200) is None


def test_get_cached_none_when_stale(feed):
    df = DataFetcher.klines_to_df(_synthetic_rows(200))
    feed.ingest("ETHUSDT", df)
    feed._last_event["ETHUSDT"] = time.monotonic() - 3600  # 1h old
    assert feed.get_cached("ETHUSDT", 200) is None


def test_get_cached_none_when_feed_disabled(feed, monkeypatch):
    monkeypatch.setattr(settings, "USE_WS_FEED", False)
    df = DataFetcher.klines_to_df(_synthetic_rows(200))
    feed.ingest("ETHUSDT", df)
    assert feed.get_cached("ETHUSDT", 200) is None


def test_get_cached_none_when_not_started(feed):
    df = DataFetcher.klines_to_df(_synthetic_rows(200))
    feed.ingest("ETHUSDT", df)
    assert feed.get_cached("ETHUSDT", 200) is None  # _started False


# ---------- data_fetcher integration ----------

def test_get_candles_prefers_ws_cache(monkeypatch):
    df = DataFetcher.klines_to_df(_synthetic_rows(200))
    ws_feed.ingest("BTCUSDT", df)
    ws_feed._started = True
    ws_feed._last_event["BTCUSDT"] = time.monotonic()
    try:
        def _boom(*a, **k):
            raise AssertionError("REST must not be called when the WS cache is fresh")
        monkeypatch.setattr(
            __import__("src.core.binance_client", fromlist=["binance_client"]).binance_client,
            "get_klines", _boom)
        out = data_fetcher.get_candles("BTCUSDT", "1h", 200)
        assert len(out) == 200
        assert float(out["close"].iloc[-1]) == pytest.approx(df["close"].iloc[-1])
    finally:
        ws_feed._started = False  # never leave the singleton "live"


def test_get_candles_rest_fallback_ingests_cache(monkeypatch):
    ws_feed._bars.pop("SOLUSDT", None)
    ws_feed._last_event.pop("SOLUSDT", None)
    rows = _synthetic_rows(200)

    class _FakeClient:
        def get_klines(self, symbol, interval, limit=200, **kw):
            return rows

    # NOTE: "import src.core.data_fetcher as x" yields the data_fetcher
    # INSTANCE (src/core/__init__.py re-export shadowing) - use importlib
    import importlib
    dfmod = importlib.import_module("src.core.data_fetcher")
    monkeypatch.setattr(dfmod, "binance_client", _FakeClient())
    out = data_fetcher.get_candles("SOLUSDT", "1h", 200)
    assert len(out) == 200
    # REST fetch re-seeds the WS cache -> next cycle reads free
    assert len(ws_feed._bars.get("SOLUSDT|1h", [])) == 200
    assert ws_feed._bars["SOLUSDT|1h"][-1][4] == rows[-1][4]


def test_non_1h_intervals_skip_cache(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("15m must not consult the 1h WS cache")
    # fresh 1h cache exists, but 15m must go straight to REST
    ws_feed.ingest("BNBUSDT", DataFetcher.klines_to_df(_synthetic_rows(200)))
    ws_feed._started = True
    ws_feed._last_event["BNBUSDT"] = time.monotonic()
    try:
        monkeypatch.setattr(
            __import__("src.core.binance_client", fromlist=["binance_client"]).binance_client,
            "get_klines", _boom)
        with pytest.raises(AssertionError):
            data_fetcher.get_candles("BNBUSDT", "15m", 200)
    finally:
        ws_feed._started = False


# ---------- reconnect / reseed safety ----------

def test_long_gap_flags_reseed(feed):
    df = DataFetcher.klines_to_df(_synthetic_rows(200))
    feed.ingest("BTCUSDT", df)
    feed._last_event["BTCUSDT|1h"] = time.monotonic() - 600  # 10 min gap
    feed._on_open()
    assert feed._needs_reseed is True


def test_short_gap_no_reseed(feed):
    df = DataFetcher.klines_to_df(_synthetic_rows(200))
    feed.ingest("BTCUSDT", df)
    feed._last_event["BTCUSDT|1h"] = time.monotonic() - 5
    feed._on_open()
    assert feed._needs_reseed is False


def test_reseed_bypasses_cache(monkeypatch, feed):
    """Reseed must hit REST directly even when the (gapped) cache is fresh."""
    calls = []
    monkeypatch.setattr(
        DataFetcher, "_get_candles_rest",
        staticmethod(lambda sym, tf="1h", lim=200: calls.append(sym) or
                     DataFetcher.klines_to_df(_synthetic_rows(200))))
    df = DataFetcher.klines_to_df(_synthetic_rows(200))
    feed.ingest("BTCUSDT", df)
    feed.ingest("ETHUSDT", df)
    feed._reseed_all()
    assert set(calls) == {"BTCUSDT", "ETHUSDT"}
    assert feed._needs_reseed is False


def test_update_universe_reconnects_when_changed(feed):
    feed._started = True
    feed.update_universe(["BTCUSDT", "ETHUSDT"])
    assert feed._universe == ["BTCUSDT", "ETHUSDT"]
    assert feed._needs_reseed is True
    feed._started = False


# ---------- status ----------

def test_status_shape(feed):
    feed.ingest("BTCUSDT", DataFetcher.klines_to_df(_synthetic_rows(10)))
    s = feed.status()
    assert s["enabled"] == settings.USE_WS_FEED
    assert s["cached_symbols"] == 1
    assert "connected" in s and "last_msg_age_s" in s


# ---------- analyzer symbol-list cache ----------

def test_refresh_symbols_cached_within_window(monkeypatch):
    from src.analysis.analyzer import analyzer
    from src.utils.helpers import now_utc
    monkeypatch.setattr(settings, "USE_ALL_USDT_PAIRS", True)
    monkeypatch.setattr(settings, "SYMBOL_REFRESH_MIN", 60)
    calls = []
    monkeypatch.setattr(
        analyzer, "_fetch_all_usdt_pairs",
        lambda: calls.append(1) or ["BTCUSDT", "ETHUSDT"])
    analyzer._symbols_ts = now_utc()          # just refreshed
    analyzer.refresh_symbols()
    assert calls == []                        # cache hit - zero API weight
    analyzer.refresh_symbols(force=True)
    assert len(calls) == 1                    # forced refresh


def test_refresh_symbols_refetches_after_ttl(monkeypatch):
    from src.analysis.analyzer import analyzer
    from src.utils.helpers import now_utc
    from datetime import timedelta
    monkeypatch.setattr(settings, "USE_ALL_USDT_PAIRS", True)
    calls = []
    monkeypatch.setattr(
        analyzer, "_fetch_all_usdt_pairs",
        lambda: calls.append(1) or ["BTCUSDT"])
    analyzer._symbols_ts = now_utc() - timedelta(hours=2)
    analyzer.refresh_symbols()
    assert len(calls) == 1
    analyzer._symbols_ts = None               # restore default-ish state
