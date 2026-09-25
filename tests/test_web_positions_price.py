"""Regression (2026-09-24): /api/positions showed no current price / P&L.

Root cause: get_tickers_batch uses GET /api/v3/ticker/price whose rows
carry the key "price", but the endpoint read "lastPrice" (a key that only
exists on the /ticker/24hr payload). Every position silently got
current_price 0.0 -> None, and the dashboard rendered '—' for
"السعر الحالي" with P&L stuck at 0.
"""
import importlib
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient

# NOTE: src/core/__init__.py re-exports the data_fetcher INSTANCE, so
# "import src.core.data_fetcher" binds the instance, not the module.
# Patching must target the real module's global binance_client.
dfmod = importlib.import_module("src.core.data_fetcher")

from src.risk.manager import risk_manager
from src.web.app import app

client = TestClient(app)


def _fake_position():
    return {
        "symbol": "ACEUSDT",
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


def test_positions_price_from_ticker_price_payload(monkeypatch):
    """The REAL /ticker/price payload uses key 'price'. The endpoint must
    surface it as current_price (was silently zeroed via lastPrice lookup)."""
    monkeypatch.setattr(risk_manager, "open_positions", [_fake_position()])

    def fake_batch(symbols, priority=False):
        rows = [{"symbol": s, "price": "1.05"} for s in symbols]
        return {r["symbol"]: r for r in rows}

    monkeypatch.setattr(dfmod.binance_client, "get_tickers_batch", fake_batch)

    r = client.get("/api/positions")
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 1
    pos = body[0]
    assert pos["current_price"] == 1.05
    # price above entry (1.05 vs 1.00) -> net P&L must be positive after fees
    assert pos["current_pnl"] > 0
    assert pos["current_pnl_pct"] > 0


def test_get_batch_prices_handles_both_key_shapes(monkeypatch):
    """get_batch_prices must accept /ticker/price rows ('price') AND
    legacy /ticker/24hr rows ('lastPrice')."""
    def fake_batch(symbols, priority=False):
        return {
            "ACEUSDT": {"symbol": "ACEUSDT", "price": "1.05"},      # /ticker/price
            "LTCUSDT": {"symbol": "LTCUSDT", "lastPrice": "88.5"},  # /ticker/24hr
        }

    monkeypatch.setattr(dfmod.binance_client, "get_tickers_batch", fake_batch)
    out = dfmod.DataFetcher.get_batch_prices(["ACEUSDT", "LTCUSDT"])
    assert out == {"ACEUSDT": 1.05, "LTCUSDT": 88.5}


def test_positions_without_price_degrades_gracefully(monkeypatch):
    """If ALL price fetches fail (batch + full-market fallback), positions
    must still be returned (current_price None, P&L zeroed) - no 500 error."""
    monkeypatch.setattr(risk_manager, "open_positions", [_fake_position()])

    def failing(*a, **k):
        raise ConnectionError("binance down")

    monkeypatch.setattr(dfmod.binance_client, "get_tickers_batch", failing)
    monkeypatch.setattr(dfmod.binance_client, "get_all_prices", failing)
    monkeypatch.setattr(dfmod.binance_client, "get_all_tickers", failing)
    # no last-known prices cached either (fresh test state)
    monkeypatch.setattr(dfmod.DataFetcher, "_LAST_PRICES", {})

    r = client.get("/api/positions")
    assert r.status_code == 200
    pos = r.json()[0]
    assert pos["current_price"] is None
    assert pos["current_pnl"] == 0
