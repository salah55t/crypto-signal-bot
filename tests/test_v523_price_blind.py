"""v5.23: position prices never blind.

Production 2026-10-01: SPCXBUSDT/CRCLBUSDT were opened by the bottom
channel but are NOT in the analyzer universe, so their @miniTicker WS
streams were never subscribed. Every /api/positions poll fell back to
REST, which the shared-IP cooldown blocked ("Rate budget exhausted (2w/4w)")
-> dashboard P&L frozen at 0 AND the 1-min SL/TP monitor blind for exactly
those positions. Fixes under test:

  1. ws_feed.set_pinned() - position/pending symbols survive analyzer
     universe refreshes (pinned into the stream universe).
  2. get_batch_tickers re-raises RateLimitError instead of "falling back"
     to the heavier weight-4 full list (guaranteed to fail too).
  3. get_batch_prices skips REST entirely during a cooldown (no 90s
     blocking acquire, no double error spam).
  4. Last-known prices persist to data/last_prices.json and load lazily,
     so a fresh process that boots inside a restored cooldown can still
     price open positions.
  5. run_position_watch pins positions+pendings every minute.
  6. /api/positions combines live + last-known prices and never blocks
     the event loop on the rate limiter.
"""
import importlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

wsmod = importlib.import_module("src.core.ws_feed")
dfmod = importlib.import_module("src.core.data_fetcher")
rlmod = importlib.import_module("src.core.rate_limiter")
cycmod = importlib.import_module("src.core.cycle")

from src.core.ws_feed import WSKlineFeed
from src.core.data_fetcher import DataFetcher
from src.core.rate_limiter import RateLimitError
from src.web.app import app
from src.risk.manager import risk_manager

from fastapi.testclient import TestClient


def _pos(symbol):
    return {
        "symbol": symbol,
        "direction": "bullish",
        "entry_price": 1.0,
        "stop_loss": 0.95,
        "take_profit": 1.10,
        "notional_usd": 100.0,
        "size": 100.0,
        "entry_fee": 0.1,
        "entry_time": datetime.now(timezone.utc).isoformat(),
        "risk_updates": [],
        "partial_closes": [],
        "paper": 1,
    }


# ---------------------------------------------------------------------------
# 1) WS pinning
# ---------------------------------------------------------------------------

def test_set_pinned_merges_into_universe_and_survives_refresh():
    feed = WSKlineFeed()
    feed._universe = ["BTCUSDT", "ETHUSDT"]
    feed.set_pinned(["spcxbusdt ", "CRCLBUSDT"])  # messy casing must be normalized
    assert set(feed._universe) == {
        "BTCUSDT", "ETHUSDT", "SPCXBUSDT", "CRCLBUSDT"}
    # an analyzer universe refresh that drops the position symbols must
    # NOT unpin them - the merged set keeps them subscribed
    feed.update_universe(["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    assert "SPCXBUSDT" in feed._universe
    assert "CRCLBUSDT" in feed._universe
    assert "SOLUSDT" in feed._universe


def test_set_pinned_idempotent_and_reconnect_only_on_change(monkeypatch):
    feed = WSKlineFeed()
    feed._started = True
    feed._universe = ["BTCUSDT", "SPCXBUSDT"]
    calls = []
    monkeypatch.setattr(feed, "_restart", lambda: calls.append(1))

    feed.set_pinned(["SPCXBUSDT"])   # pin already covered by universe
    assert calls == []               # merged set identical -> no reconnect
    feed.set_pinned(["SPCXBUSDT"])   # identical pin set -> no-op
    assert calls == []
    feed.set_pinned(["SPCXBUSDT", "CRCLBUSDT"])  # NEW pin -> one reconnect
    assert len(calls) == 1
    assert "CRCLBUSDT" in feed._universe


def test_update_universe_no_reconnect_when_merged_stable(monkeypatch):
    """Analyzer churn + identical pins must not cause reconnect storms."""
    feed = WSKlineFeed()
    feed._universe = ["BTCUSDT", "SPCXBUSDT"]
    feed._pinned = {"SPCXBUSDT"}
    feed._started = True
    calls = []
    monkeypatch.setattr(feed, "_restart", lambda: calls.append(1))
    feed.update_universe(["BTCUSDT"])  # effective set unchanged
    assert calls == []
    feed.update_universe(["BTCUSDT", "ETHUSDT"])  # real change -> reconnect
    assert len(calls) == 1
    assert "SPCXBUSDT" in feed._universe  # pin survived


# ---------------------------------------------------------------------------
# 2) REST behaviour
# ---------------------------------------------------------------------------

def test_batch_tickers_reraises_rate_limit(monkeypatch):
    """Budget exhaustion must NOT trigger the weight-4 full-list fallback:
    it costs MORE than the 2w batch and fails for the same reason."""
    def boom(*a, **k):
        raise RateLimitError("Rate budget exhausted (2w needed)")

    def must_not_run(*a, **k):
        raise AssertionError("weight-4 full list ran after a rate-limit error")

    monkeypatch.setattr(dfmod.binance_client, "get_tickers_batch", boom)
    monkeypatch.setattr(dfmod.binance_client, "get_all_prices", must_not_run)
    with pytest.raises(RateLimitError):
        dfmod.data_fetcher.get_batch_tickers(["SPCXBUSDT"])


def test_batch_tickers_still_falls_back_on_transport_error(monkeypatch):
    """Non-budget failures keep the full-list fallback (it CAN recover)."""
    def boom(*a, **k):
        raise ConnectionError("transient")

    monkeypatch.setattr(dfmod.binance_client, "get_tickers_batch", boom)
    monkeypatch.setattr(
        dfmod.binance_client, "get_all_prices",
        lambda *a, **k: [{"symbol": "SPCXBUSDT", "price": "151.2"}])
    out = dfmod.data_fetcher.get_batch_tickers(["SPCXBUSDT"])
    assert out["SPCXBUSDT"]["price"] == "151.2"


def test_get_batch_prices_skips_rest_during_cooldown(monkeypatch):
    """During a cooldown REST is doomed - get_batch_prices must return the
    WS-only result immediately instead of blocking ~90s on a refusal."""
    monkeypatch.setattr(
        dfmod.DataFetcher, "_LAST_PRICES",
        {"CRCLBUSDT": (time.time(), 82.66)})
    monkeypatch.setattr(wsmod.ws_feed, "get_live_prices",
                        lambda syms, max_age_s=None: {})
    monkeypatch.setattr(rlmod.rate_limiter, "cooldown_remaining",
                        lambda: 300.0)

    def must_not_run(*a, **k):
        raise AssertionError("REST touched during an active cooldown")

    monkeypatch.setattr(dfmod.binance_client, "get_tickers_batch", must_not_run)
    monkeypatch.setattr(dfmod.binance_client, "get_all_prices", must_not_run)
    out = DataFetcher.get_batch_prices(["CRCLBUSDT"], priority=True)
    assert out == {}  # partial result; callers add last-known prices


def test_get_batch_prices_serves_ws_prices_without_rest(monkeypatch):
    monkeypatch.setattr(dfmod.DataFetcher, "_LAST_PRICES", {})
    monkeypatch.setattr(wsmod.ws_feed, "get_live_prices",
                        lambda syms, max_age_s=None:
                        {"SPCXBUSDT": 151.2})
    monkeypatch.setattr(rlmod.rate_limiter, "cooldown_remaining", lambda: 0.0)

    def must_not_run(*a, **k):
        raise AssertionError("REST used although WS covered every symbol")

    monkeypatch.setattr(dfmod.binance_client, "get_tickers_batch", must_not_run)
    out = DataFetcher.get_batch_prices(["SPCXBUSDT"], priority=True)
    assert out == {"SPCXBUSDT": 151.2}
    # and the fresh WS price must land in the last-known table
    assert DataFetcher._LAST_PRICES["SPCXBUSDT"][1] == 151.2


# ---------------------------------------------------------------------------
# 3) disk-backed last-known prices
# ---------------------------------------------------------------------------

def test_last_prices_disk_roundtrip(monkeypatch, tmp_path):
    f = tmp_path / "last_prices.json"
    monkeypatch.setattr(DataFetcher, "_LAST_PRICES_FILE", f)
    monkeypatch.setattr(DataFetcher, "_LAST_PRICES", {})
    monkeypatch.setattr(DataFetcher, "_LAST_PRICES_LAST_SAVE", 0.0)
    monkeypatch.setattr(DataFetcher, "_LAST_PRICES_DISK_LOADED", False)
    monkeypatch.setattr(wsmod.ws_feed, "get_live_prices",
                        lambda syms, max_age_s=None: {})
    monkeypatch.setattr(rlmod.rate_limiter, "cooldown_remaining", lambda: 0.0)
    monkeypatch.setattr(
        dfmod.binance_client, "get_tickers_batch",
        lambda syms, priority=False: {s: {"price": "82.66"} for s in syms})

    out = DataFetcher.get_batch_prices(["CRCLBUSDT"])
    assert out == {"CRCLBUSDT": 82.66}
    on_disk = json.loads(f.read_text())
    assert on_disk["CRCLBUSDT"]["price"] == 82.66

    # simulate a fresh process: memory wiped, disk flag reset
    monkeypatch.setattr(DataFetcher, "_LAST_PRICES", {})
    monkeypatch.setattr(DataFetcher, "_LAST_PRICES_DISK_LOADED", False)
    known = DataFetcher.get_last_known_prices(["CRCLBUSDT"], max_age_s=900)
    assert known == {"CRCLBUSDT": 82.66}

    # stale disk rows must be filtered by max_age
    f.write_text(json.dumps({
        "CRCLBUSDT": {"price": 82.66, "ts": time.time() - 300}}))
    monkeypatch.setattr(DataFetcher, "_LAST_PRICES", {})
    monkeypatch.setattr(DataFetcher, "_LAST_PRICES_DISK_LOADED", False)
    assert DataFetcher.get_last_known_prices(
        ["CRCLBUSDT"], max_age_s=60) == {}


def test_last_prices_memory_wins_over_disk(monkeypatch, tmp_path):
    f = tmp_path / "last_prices.json"
    f.write_text(json.dumps({
        "CRCLBUSDT": {"price": 1.0, "ts": time.time()}}))
    monkeypatch.setattr(DataFetcher, "_LAST_PRICES_FILE", f)
    monkeypatch.setattr(DataFetcher, "_LAST_PRICES",
                        {"CRCLBUSDT": (time.time(), 82.66)})
    monkeypatch.setattr(DataFetcher, "_LAST_PRICES_DISK_LOADED", False)
    known = DataFetcher.get_last_known_prices(["CRCLBUSDT"])
    assert known == {"CRCLBUSDT": 82.66}  # fresher memory entry kept


# ---------------------------------------------------------------------------
# 4) wiring: watcher + dashboard endpoint
# ---------------------------------------------------------------------------

def test_position_watch_pins_positions_and_pendings(monkeypatch):
    pinned = set()

    def fake_pin(syms):
        pinned.update(s.upper() for s in syms)

    monkeypatch.setattr(wsmod.ws_feed, "set_pinned", fake_pin)
    monkeypatch.setattr(cycmod.risk_manager, "open_positions",
                        [_pos("SPCXBUSDT")])
    monkeypatch.setattr(cycmod.risk_manager, "pending_entries",
                        [{"symbol": "CRCLBUSDT"}])
    monkeypatch.setattr(cycmod, "fetch_prices", lambda syms, priority=False: {})
    monkeypatch.setattr(dfmod.DataFetcher, "get_last_known_prices",
                        lambda syms, max_age_s=120: {})
    cycmod.run_position_watch()
    assert pinned == {"SPCXBUSDT", "CRCLBUSDT"}


def test_positions_endpoint_serves_last_known_during_cooldown(monkeypatch):
    """The exact production incident: cooldown active + WS blind on the
    position symbol -> the endpoint must still price the position from
    last-known data instead of returning current_price=None."""
    monkeypatch.setattr(risk_manager, "open_positions", [_pos("SPCXBUSDT")])
    monkeypatch.setattr(wsmod.ws_feed, "set_pinned", lambda syms: None)
    monkeypatch.setattr(wsmod.ws_feed, "get_live_prices",
                        lambda syms, max_age_s=None: {})
    monkeypatch.setattr(rlmod.rate_limiter, "cooldown_remaining",
                        lambda: 300.0)

    def must_not_run(*a, **k):
        raise AssertionError("REST touched during an active cooldown")

    monkeypatch.setattr(dfmod.binance_client, "get_tickers_batch", must_not_run)
    monkeypatch.setattr(dfmod.binance_client, "get_all_prices", must_not_run)
    monkeypatch.setattr(
        dfmod.DataFetcher, "_LAST_PRICES",
        {"SPCXBUSDT": (time.time(), 151.2)})
    monkeypatch.setattr(DataFetcher, "_LAST_PRICES_DISK_LOADED", True)

    client = TestClient(app)
    r = client.get("/api/positions")
    assert r.status_code == 200
    pos = r.json()[0]
    assert pos["current_price"] == 151.2
    # entry 1.0 -> current 151.2 must produce a strongly positive P&L
    assert pos["current_pnl"] > 0


def test_positions_endpoint_pins_position_symbols(monkeypatch):
    monkeypatch.setattr(risk_manager, "open_positions", [_pos("CRCLBUSDT")])
    pinned = set()

    def fake_pin(syms):
        pinned.update(s.upper() for s in syms)

    monkeypatch.setattr(wsmod.ws_feed, "set_pinned", fake_pin)
    monkeypatch.setattr(wsmod.ws_feed, "get_live_prices",
                        lambda syms, max_age_s=None:
                        {"CRCLBUSDT": 82.66})
    monkeypatch.setattr(rlmod.rate_limiter, "cooldown_remaining", lambda: 0.0)

    client = TestClient(app)
    r = client.get("/api/positions")
    assert r.status_code == 200
    assert pinned == {"CRCLBUSDT"}
    assert r.json()[0]["current_price"] == 82.66
