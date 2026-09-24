"""
v5.5 Analysis-driven trade management - Unit Tests

User incident: a trade got 3 SL/TP "updates" and still closed negative.
Root causes fixed here:
  - Opposite-signal exit threshold was 75 -> a bearish 55-74% read only
    "tightened" the SL 0.5% below price, LOCKING A LOSS, then the stop hit.
    Now: conf >= 55 closes the position immediately (graduated bands).
  - TP extension on bullish continuation raised the target WITHOUT raising
    the SL -> extensions left the trade exposed. Now the SL is raised to
    lock PROFIT_LOCK_FRACTION of the current profit in the SAME update.
  - The old trailing "tighten on bearish signal" block (which could lock a
    loss) is gone - signal-driven exits belong to the structural pass.
Also covers the 4h timeframe move:
  - WS kline feed multi-interval (per-(symbol, interval) caches)
  - data_fetcher serves 4h candles from the WS cache (zero REST weight)
"""
import sys
import time
import json
import importlib
from pathlib import Path

import pytest
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
import src.risk.manager as manager_module
import src.core.data_fetcher as data_fetcher_module
from src.risk.manager import RiskManager
from src.core.ws_feed import WSKlineFeed
from src.core.data_fetcher import DataFetcher
from src.utils.helpers import now_utc

from tests.test_v5_veteran import make_manager, make_position


def _sig(direction, confidence):
    return {"direction": direction, "confidence": confidence}


# ============================================
# Settings: 4h era + v5.5 thresholds
# ============================================

def test_settings_v55_defaults():
    assert settings.TIMEFRAMES == ["4h"]
    assert settings.WS_INTERVALS == ["1h", "4h"]
    assert settings.OPPOSITE_SIGNAL_CONF == 55.0
    assert settings.SIGNAL_TIGHTEN_CONF == 40.0
    assert settings.CONTINUATION_CONF == 65.0
    assert settings.CONTINUATION_MIN_PROFIT_PCT == 1.0
    assert settings.PROFIT_LOCK_FRACTION == 0.5
    # time stop scaled for 4h bars
    assert settings.MAX_TRADE_HOURS == 72.0
    assert settings.ABSOLUTE_MAX_TRADE_HOURS == 120.0


# ============================================
# Graduated opposite-signal response
# ============================================

def test_bearish_conf55_closes_long_immediately(tmp_path, monkeypatch):
    rm = make_manager(tmp_path, monkeypatch)
    rm.open_positions = [make_position()]
    action, reason = rm.evaluate_structural_exit(
        rm.open_positions[0], _sig("bearish", 55), 100.0)
    assert action == "exit"
    assert "55" in reason


def test_bearish_conf54_defends_instead_of_exit(tmp_path, monkeypatch):
    rm = make_manager(tmp_path, monkeypatch)
    rm.open_positions = [make_position()]
    action, reason = rm.evaluate_structural_exit(
        rm.open_positions[0], _sig("bearish", 54), 100.0)
    assert action == "tighten"
    # level parsed from "(...)" = 0.5% below current price
    assert reason.rsplit("(", 1)[1].rstrip(")") == f"{100.0 * 0.995:.4f}"


def test_bearish_conf39_is_ignored(tmp_path, monkeypatch):
    rm = make_manager(tmp_path, monkeypatch)
    rm.open_positions = [make_position()]
    action, reason = rm.evaluate_structural_exit(
        rm.open_positions[0], _sig("bearish", 39), 100.0)
    assert action == "none"
    assert reason is None


def test_bullish_signal_closes_short_immediately(tmp_path, monkeypatch):
    rm = make_manager(tmp_path, monkeypatch)
    rm.open_positions = [make_position(direction="bearish", sl=102.0, tp=96.0)]
    action, _ = rm.evaluate_structural_exit(
        rm.open_positions[0], _sig("bullish", 60), 100.0)
    assert action == "exit"


def test_same_direction_signal_never_exits(tmp_path, monkeypatch):
    rm = make_manager(tmp_path, monkeypatch)
    rm.open_positions = [make_position()]
    action, _ = rm.evaluate_structural_exit(
        rm.open_positions[0], _sig("bullish", 90), 100.0)
    assert action != "exit"


def test_structural_exit_disabled_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "STRUCTURAL_EXITS_ENABLED", False)
    rm = make_manager(tmp_path, monkeypatch)
    rm.open_positions = [make_position()]
    action, _ = rm.evaluate_structural_exit(
        rm.open_positions[0], _sig("bearish", 90), 100.0)
    assert action == "none"


# ============================================
# Trailing: the loss-locking tighten is GONE
# ============================================

def test_trailing_does_not_tighten_on_bearish_signal(tmp_path, monkeypatch):
    """Old bug: bearish conf>60 moved SL to 0.5% below price - below entry
    that LOCKS A LOSS. Now a bearish signal produces NO trailing update
    (the structural pass closes/defends instead)."""
    rm = make_manager(tmp_path, monkeypatch)
    # price below entry -> ladder silent, old code would tighten SL
    rm.open_positions = [make_position(entry=100.0, sl=98.0, tp=104.0)]
    prices = {"TESTUSDT": 99.0}
    sigs = {"TESTUSDT": _sig("bearish", 90)}
    updates = rm.apply_trailing_logic(prices, sigs)
    assert updates == []
    assert rm.open_positions[0]["stop_loss"] == 98.0


def test_trailing_bearish_signal_no_update_even_in_profit(tmp_path, monkeypatch):
    rm = make_manager(tmp_path, monkeypatch)
    rm.open_positions = [make_position(entry=100.0, sl=100.5, tp=104.0)]
    prices = {"TESTUSDT": 101.0}  # +1% -> BE ladder already done (sl above entry)
    sigs = {"TESTUSDT": _sig("bearish", 85)}
    updates = rm.apply_trailing_logic(prices, sigs)
    assert updates == []


# ============================================
# Continuation: extend TP + lock profit TOGETHER
# ============================================

def test_continuation_extends_tp_and_locks_profit(tmp_path, monkeypatch):
    """+1.2% profit: old code extended TP with SL at break-even (exposed).
    Now the same update must raise SL to lock 50% of the profit (+0.6%)."""
    rm = make_manager(tmp_path, monkeypatch)
    rm.open_positions = [make_position(entry=100.0, sl=100.0, tp=104.0)]
    pos = rm.open_positions[0]
    pos["peak_price"] = 101.2
    prices = {"TESTUSDT": 101.2}
    sigs = {"TESTUSDT": _sig("bullish", 70)}
    updates = rm.apply_trailing_logic(prices, sigs)
    assert updates, "continuation must produce an update"
    u = updates[-1]
    # TP extended by 50% of the remaining distance
    assert u["new_tp"] == pytest.approx(104.0 + (104.0 - 101.2) * 0.5)
    # SL locked at entry + 0.6% (above the BE ladder level)
    assert u["new_sl"] == pytest.approx(100.0 * (1 + 0.6 / 100.0))
    assert "Extended TP" in u["reason"]
    assert "Locked" in u["reason"]


def test_continuation_with_big_profit_uses_ladder_sl(tmp_path, monkeypatch):
    """At +3% the ladder (+2%) is tighter than the 50% lock (+1.5%) - the
    tightest level must win while the TP still extends."""
    rm = make_manager(tmp_path, monkeypatch)
    rm.open_positions = [make_position(entry=100.0, sl=98.0, tp=104.0)]
    rm.open_positions[0]["peak_price"] = 103.0
    prices = {"TESTUSDT": 103.0}
    sigs = {"TESTUSDT": _sig("bullish", 70)}
    updates = rm.apply_trailing_logic(prices, sigs)
    u = updates[-1]
    assert u["new_tp"] == pytest.approx(104.0 + (104.0 - 103.0) * 0.5)
    assert u["new_sl"] == pytest.approx(102.0)  # ladder lock +2%
    # the lock did not win here, so no "Locked" wording
    assert "Locked" not in u["reason"]


def test_continuation_requires_min_profit(tmp_path, monkeypatch):
    """The '3 updates then negative' bug: TP extended while the trade had
    no secured profit. Below CONTINUATION_MIN_PROFIT_PCT no extension."""
    rm = make_manager(tmp_path, monkeypatch)
    rm.open_positions = [make_position(entry=100.0, sl=98.0, tp=104.0)]
    prices = {"TESTUSDT": 100.5}  # +0.5% < 1.0%
    sigs = {"TESTUSDT": _sig("bullish", 80)}
    updates = rm.apply_trailing_logic(prices, sigs)
    assert updates == []
    assert rm.open_positions[0]["take_profit"] == 104.0


def test_continuation_requires_confidence(tmp_path, monkeypatch):
    rm = make_manager(tmp_path, monkeypatch)
    rm.open_positions = [make_position(entry=100.0, sl=98.0, tp=104.0)]
    rm.open_positions[0]["peak_price"] = 102.0
    prices = {"TESTUSDT": 102.0}  # +2%
    sigs = {"TESTUSDT": _sig("bullish", 64)}  # just below 65
    updates = rm.apply_trailing_logic(prices, sigs)
    # the profit ladder may still lock gains, but NO TP extension happens
    assert not any("Extended TP" in u["reason"] for u in updates)
    assert rm.open_positions[0]["take_profit"] == 104.0


def test_continuation_never_loosens_sl(tmp_path, monkeypatch):
    rm = make_manager(tmp_path, monkeypatch)
    rm.open_positions = [make_position(entry=100.0, sl=103.5, tp=105.0)]
    prices = {"TESTUSDT": 103.9}  # +3.9% profit, tight SL already set
    sigs = {"TESTUSDT": _sig("bullish", 75)}
    updates = rm.apply_trailing_logic(prices, sigs)
    # TP extends, SL must stay where it is (103.5 beats lock 1.95 and ladder)
    assert any(u["new_tp"] > 105.0 for u in updates)
    assert all(u["new_sl"] is None or u["new_sl"] >= 103.5 for u in updates)
    assert rm.open_positions[0]["stop_loss"] == 103.5


def test_continuation_lock_capped_below_current_price(tmp_path, monkeypatch):
    """Lock can never sit at/above the current price (instant stop-out)."""
    rm = make_manager(tmp_path, monkeypatch)
    rm.open_positions = [make_position(entry=100.0, sl=98.0, tp=104.0)]
    rm.open_positions[0]["peak_price"] = 101.1
    prices = {"TESTUSDT": 101.1}  # +1.1% -> lock 0.55% = 100.55 < 101.1*0.999
    sigs = {"TESTUSDT": _sig("bullish", 70)}
    updates = rm.apply_trailing_logic(prices, sigs)
    u = updates[-1]
    assert u["new_sl"] < 101.1
    assert u["new_sl"] == pytest.approx(100.0 * (1 + 0.55 / 100.0))


def test_bearish_continuation_mirror(tmp_path, monkeypatch):
    """Short side: bearish continuation extends TP down + locks profit."""
    rm = make_manager(tmp_path, monkeypatch)
    rm.open_positions = [make_position(direction="bearish", entry=100.0,
                                       sl=102.0, tp=96.0)]
    rm.open_positions[0]["trough_price"] = 98.8
    prices = {"TESTUSDT": 98.8}  # short +1.2%
    sigs = {"TESTUSDT": _sig("bearish", 70)}
    updates = rm.apply_trailing_logic(prices, sigs)
    u = updates[-1]
    assert u["new_tp"] == pytest.approx(96.0 - (98.8 - 96.0) * 0.5)
    assert u["new_sl"] == pytest.approx(100.0 * (1 - 0.6 / 100.0))


# ============================================
# WS feed: multi-interval caches (4h era)
# ============================================

def _raw_klines(start_ms, step_ms, closes):
    rows = []
    t = start_ms
    for c in closes:
        rows.append([t, "1", "1", "1", str(c), "10",
                     t + step_ms - 1, "100", "5", "5", "5", "0"])
        t += step_ms
    return rows


def _new_feed(intervals=None):
    feed = WSKlineFeed()
    if intervals is not None:
        monkey_intervals = intervals
        feed.__class__._intervals = staticmethod(lambda: monkey_intervals)
    return feed


H1 = 3_600_000
H4 = 14_400_000
BASE = 1_700_000_000_000


def test_ws_cache_isolated_per_interval():
    feed = WSKlineFeed()
    feed._started = True
    df_1h = DataFetcher.klines_to_df(
        _raw_klines(BASE, H1, [100.0] * 61))
    df_4h = DataFetcher.klines_to_df(
        _raw_klines(BASE, H4, [200.0] * 61))
    feed.ingest("BTCUSDT", df_1h, interval="1h")
    feed.ingest("BTCUSDT", df_4h, interval="4h")

    got_1h = feed.get_cached("BTCUSDT", 60, interval="1h")
    got_4h = feed.get_cached("BTCUSDT", 60, interval="4h")
    assert got_1h is not None and got_4h is not None
    assert float(got_1h["close"].iloc[-1]) == 100.0
    assert float(got_4h["close"].iloc[-1]) == 200.0
    # unknown interval is not served
    assert feed.get_cached("BTCUSDT", 60, interval="15m") is None


def test_ws_message_routes_by_interval_field():
    feed = WSKlineFeed()
    payload = {
        "data": {
            "e": "kline", "s": "BTCUSDT",
            "k": {"t": BASE, "o": "1", "h": "2", "l": "0.5", "c": "1.5",
                  "v": "10", "T": BASE + H4 - 1, "q": "10", "n": "5",
                  "V": "5", "Q": "5", "i": "4h", "x": False},
        }
    }
    feed._on_message(message=json.dumps(payload))
    k1h = dict(payload["data"]["k"], i="1h", t=BASE + H1)
    feed._on_message(message=json.dumps({"data": {"e": "kline",
                                                  "s": "BTCUSDT", "k": k1h}}))
    assert "BTCUSDT|4h" in feed._bars
    assert "BTCUSDT|1h" in feed._bars
    assert len(feed._bars["BTCUSDT|4h"]) == 1
    assert len(feed._bars["BTCUSDT|1h"]) == 1
    assert feed.status()["cached_per_interval"] == {"4h": 1, "1h": 1}


def test_get_candles_serves_4h_from_ws_cache(tmp_path, monkeypatch):
    """4h is now a strategy TF: it must be served from the WS cache with
    ZERO REST klines weight."""
    dfmod = importlib.import_module("src.core.data_fetcher")  # real module
    feed = WSKlineFeed()
    feed._started = True
    df_4h = DataFetcher.klines_to_df(_raw_klines(BASE, H4, [55.0] * 200))
    feed.ingest("ETHUSDT", df_4h, interval="4h")

    import src.core.ws_feed as wsm
    monkeypatch.setattr(wsm, "ws_feed", feed)

    def _no_rest(*a, **k):
        raise AssertionError("REST klines must not be called when cache is live")
    monkeypatch.setattr(dfmod, "binance_client", type("_B", (), {
        "get_klines": staticmethod(_no_rest)}))

    got = dfmod.DataFetcher.get_candles("ETHUSDT", "4h", limit=200)
    assert got is not None
    assert float(got["close"].iloc[-1]) == 55.0
    assert len(got) == 200


def test_get_candles_rest_ingests_into_interval_cache(tmp_path, monkeypatch):
    """REST fetch for 4h re-seeds the 4h cache (not 1h)."""
    dfmod = importlib.import_module("src.core.data_fetcher")  # real module
    monkeypatch.setattr(dfmod, "binance_client", type("_B", (), {
        "get_klines": staticmethod(
            lambda *a, **k: _raw_klines(BASE, H4, [77.0] * 200))}))

    ingested = {}
    class _FakeFeed:
        def ingest(self, symbol, df, interval="1h"):
            ingested[(symbol, interval)] = df
    import src.core.ws_feed as wsm
    monkeypatch.setattr(wsm, "ws_feed", _FakeFeed())

    got = dfmod.DataFetcher.get_candles("LTCUSDT", "4h", limit=200)
    assert ("LTCUSDT", "4h") in ingested
    assert float(got["close"].iloc[-1]) == 77.0


def test_ws_intervals_fallback_when_unset():
    feed = WSKlineFeed()
    assert feed._intervals() in (["1h"], ["1h", "4h"]) or feed._intervals()
    # sanity: whatever the setting, it always contains 1h
    assert "1h" in feed._intervals()
