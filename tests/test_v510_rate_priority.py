"""
v5.10 rate-priority + last-known prices tests.

Fixes the logged failure chain:
  WARNING Batch ticker fetch failed (Rate budget exhausted (2w needed))
  ERROR   Failed to fetch prices for positions: Rate budget exhausted (80w)

Root causes fixed:
  1. Bulk analysis burst could fill the whole 4500w/min budget, starving the
     tiny (2w) position-monitoring requests -> PRIORITY_RESERVE lane.
  2. The fallback for a failed 2w batch was a full-market /ticker/24hr call
     at weight 80 - MORE expensive and guaranteed to fail in the same
     state -> now a full /ticker/price list at weight 4.
  3. When even the fallback is blocked (shared-IP pressure cooldown) the
     dashboard /api/positions returned nothing -> serves last-known prices
     (<= 15 min) with a clear log line.

Covers:
  - rate limiter: reserve lane semantics, priority bypass of the reserve,
    priority still respects hard cooldowns
  - data_fetcher: fallback chain (never /ticker/24hr), last-known cache
  - cycle.fetch_prices: priority passthrough
  - /api/positions: last-known serving on live failure
"""
import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.rate_limiter import WeightedRateLimiter, RateLimitError

dfmod = importlib.import_module("src.core.data_fetcher")  # the MODULE
from src.core.binance_client import binance_client  # noqa: E402


# ============================================
# helpers
# ============================================

def make_limiter(budget=100.0, reserve=20.0):
    return WeightedRateLimiter(budget_per_min=budget,
                               hard_ceiling=6000,
                               priority_reserve=reserve)


@pytest.fixture(autouse=True)
def clean_last_known():
    dfmod.DataFetcher._LAST_PRICES.clear()
    yield
    dfmod.DataFetcher._LAST_PRICES.clear()


# ============================================
# priority reserve lane
# ============================================

def test_bulk_traffic_stops_at_reserve():
    rl = make_limiter(budget=100, reserve=20)  # bulk cap = 80
    assert rl.acquire(70, timeout=0.1) is True          # 70 <= 80 OK
    assert rl.acquire(15, timeout=0.1) is False         # 85 > 80 blocked
    assert rl.used_weight() == 70


def test_priority_uses_the_reserve():
    rl = make_limiter(budget=100, reserve=20)
    assert rl.acquire(70, timeout=0.1) is True
    # priority request may use 70..100 (the reserve zone)
    assert rl.acquire(15, timeout=0.1, priority=True) is True
    assert rl.used_weight() == 85


def test_priority_cannot_exceed_full_budget():
    rl = make_limiter(budget=100, reserve=20)
    assert rl.acquire(95, timeout=0.1, priority=True) is True
    assert rl.acquire(10, timeout=0.1, priority=True) is False  # 105 > 100


def test_priority_still_respects_cooldown():
    rl = make_limiter(budget=4500, reserve=200)
    rl.trigger_cooldown(60)
    # a long cooldown must reject even priority requests immediately
    assert rl.acquire(2, timeout=0.5, priority=True) is False


def test_zero_reserve_keeps_old_behaviour():
    rl = make_limiter(budget=100, reserve=0)
    assert rl.acquire(95, timeout=0.1) is True
    assert rl.acquire(10, timeout=0.1) is False  # 105 > 100, no lane


# ============================================
# data_fetcher fallback chain
# ============================================

def test_fallback_uses_weight4_price_list_never_24hr(monkeypatch):
    def boom(*a, **k):
        raise RateLimitError("Rate budget exhausted (2w needed)")
    monkeypatch.setattr(binance_client, "get_tickers_batch", boom)

    def must_not_call(*a, **k):
        raise AssertionError("/ticker/24hr (80w) must not be used for prices")
    monkeypatch.setattr(binance_client, "get_all_tickers", must_not_call)

    def fake_all_prices(priority=False):
        assert priority is True  # fallback must ride the priority lane
        return [
            {"symbol": "AAAUSDT", "price": "1.0"},
            {"symbol": "BBBUSDT", "price": "2.0"},
            {"symbol": "OTHER", "price": "9.9"},
        ]
    monkeypatch.setattr(binance_client, "get_all_prices", fake_all_prices)

    out = dfmod.DataFetcher.get_batch_tickers(["AAAUSDT", "BBBUSDT"],
                                              priority=True)
    assert set(out) == {"AAAUSDT", "BBBUSDT"}


def test_batch_success_does_not_hit_fallback(monkeypatch):
    called = {"n": 0}

    def fake_batch(symbols, priority=False):
        called["n"] += 1
        return {s: {"price": "3.0"} for s in symbols}
    monkeypatch.setattr(binance_client, "get_tickers_batch", fake_batch)

    out = dfmod.DataFetcher.get_batch_prices(["CCCUSDT"], priority=True)
    assert out == {"CCCUSDT": 3.0}
    assert called["n"] == 1
    assert hasattr(binance_client, "get_all_prices")  # new fallback exists


# ============================================
# last-known price cache
# ============================================

def test_last_known_cache_roundtrip(monkeypatch):
    monkeypatch.setattr(
        binance_client, "get_tickers_batch",
        lambda s, priority=False: {"DDDUSDT": {"price": "5.5"},
                                   "ZERO": {"price": "0"}})
    out = dfmod.DataFetcher.get_batch_prices(["DDDUSDT", "ZERO"])
    assert out == {"DDDUSDT": 5.5}  # zero price ignored + not cached
    assert dfmod.DataFetcher.get_last_known_prices(
        ["DDDUSDT", "ZERO"], max_age_s=60) == {"DDDUSDT": 5.5}


def test_last_known_respects_max_age():
    import time
    dfmod.DataFetcher._LAST_PRICES["OLDUSDT"] = (time.time() - 3600, 42.0)
    assert dfmod.DataFetcher.get_last_known_prices(
        ["OLDUSDT"], max_age_s=900) == {}
    assert dfmod.DataFetcher.get_last_known_prices(
        ["OLDUSDT"], max_age_s=7200) == {"OLDUSDT": 42.0}


def test_last_known_empty_when_never_fetched():
    assert dfmod.DataFetcher.get_last_known_prices(
        ["NEVERSEENUSDT"], max_age_s=900) == {}


# ============================================
# cycle.fetch_prices priority passthrough
# ============================================

def test_fetch_prices_passes_priority(monkeypatch):
    from src.core import cycle
    seen = {}

    def fake_get_batch_prices(symbols, priority=False):
        seen["priority"] = priority
        return {"XUSDT": 1.0}
    monkeypatch.setattr(cycle.data_fetcher, "get_batch_prices",
                        fake_get_batch_prices)
    assert cycle.fetch_prices(["XUSDT"], priority=True) == {"XUSDT": 1.0}
    assert seen["priority"] is True


def test_fetch_prices_returns_empty_on_error(monkeypatch):
    from src.core import cycle

    def boom(symbols, priority=False):
        raise RateLimitError("no budget")
    monkeypatch.setattr(cycle.data_fetcher, "get_batch_prices", boom)
    assert cycle.fetch_prices(["XUSDT"], priority=True) == {}


# ============================================
# /api/positions - last-known serving
# ============================================

def _client():
    from fastapi.testclient import TestClient
    from src.web.app import app
    return TestClient(app)


POSITION = {
    "symbol": "BTCUSDT",
    "direction": "bullish",
    "entry_price": 60000.0,
    "stop_loss": 58000.0,
    "take_profit": 66000.0,
    "notional_usd": 10.0,
    "entry_fee": 0.01,
    "size": 10.0 / 60000.0,
}


def test_api_positions_serves_last_known_on_live_failure(monkeypatch):
    from src.risk import manager as mgr_mod
    monkeypatch.setattr(mgr_mod.risk_manager, "open_positions", [POSITION])

    def boom(symbols, priority=False):
        assert priority is True
        raise RateLimitError("Rate budget exhausted (2w needed)")
    monkeypatch.setattr(dfmod.data_fetcher, "get_batch_prices", boom)
    monkeypatch.setattr(
        dfmod.data_fetcher, "get_last_known_prices",
        lambda s, max_age_s=900: {"BTCUSDT": 65000.0})

    r = _client().get("/api/positions")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 1
    # P&L computed from the last-known price (65000 vs 60000 entry)
    assert data[0]["current_price"] == 65000.0
    assert data[0]["current_pnl"] > 0


def test_api_positions_error_without_any_known_price(monkeypatch):
    from src.risk import manager as mgr_mod
    monkeypatch.setattr(mgr_mod.risk_manager, "open_positions", [POSITION])

    def boom(symbols, priority=False):
        raise RateLimitError("Rate budget exhausted (80w needed)")
    monkeypatch.setattr(dfmod.data_fetcher, "get_batch_prices", boom)
    monkeypatch.setattr(
        dfmod.data_fetcher, "get_last_known_prices",
        lambda s, max_age_s=900: {})

    r = _client().get("/api/positions")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 1
    assert data[0]["current_price"] is None  # position listed, no price
