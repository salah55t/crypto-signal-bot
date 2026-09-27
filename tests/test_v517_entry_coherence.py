"""
v5.17 entry-geometry coherence + ws_only ban survival tests.

Dashboard forensics (2026-09-27) found three defects:

  1. WS-only degraded cycles aborted during a REST ban - analyze_one
     short-circuited every symbol on _ban_active() even though ws_only
     mode spends ZERO REST weight (candles from the WS cache, order book
     skipped). Production log showed "running WS-only cycle (coverage 98%,
     zero REST weight)" followed immediately by "Analysis burst ABORTED by
     Binance rate ban (80 symbol(s) skipped)".
  2. The momentum bypass let high-confidence limit recs open at market
     13-20% ABOVE their planned entry zone (AVAX: entry 11.17 vs zone
     8.94-9.48; GRAM: +13%) - confidence is inflated by regime/confluence
     boosts, so >= 82 fired constantly.
  3. Limit recs carried an SL computed from the price at analysis time,
     not from the (far lower) golden-pocket fill price - SOLUSDT: entry
     zone 106.9-110.4 with SL 119.3; a pending fill at the pocket would
     have been stopped out on tick 1.

Fixes covered:
  - fibonacci.compute_entry_exit: SL/RR/distances anchored to the actual
    entry (market -> current price, limit -> pocket level), both branches
  - manager.validate_recommendation: catch-all "incoherent stop" gate
  - cycle.open_new_positions: PENDING_MOMENTUM_MAX_ATR hard cap on the
    momentum bypass
  - analyzer: ws_only bursts survive a REST ban; REST bursts still abort
"""
import importlib
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.indicators.fibonacci import compute_entry_exit
from src.analysis.analyzer import MarketAnalyzer, _ban_active
from src.core.rate_limiter import WeightedRateLimiter
from src.risk.manager import RiskManager
import src.risk.manager as manager_module

cfgmod = importlib.import_module("config.settings")
anmod = importlib.import_module("src.analysis.analyzer")
cycmod = importlib.import_module("src.core.cycle")


# ============================================
# helpers
# ============================================

def make_limiter(budget=100.0, reserve=20.0):
    return WeightedRateLimiter(budget_per_min=budget,
                               hard_ceiling=6000, priority_reserve=reserve)


@pytest.fixture
def isolated_rec_file(monkeypatch, tmp_path):
    rec = tmp_path / "recommendations.json"
    rec.write_text(json.dumps({"top_recommendations": [{"symbol": "OLD"}]}),
                   encoding="utf-8")
    # NOTE: src.analysis re-exports the analyzer singleton under the module
    # name - patch the module via importlib, not the string path.
    monkeypatch.setattr(anmod, "RECOMMENDATIONS_FILE", rec)
    return rec


def _analyzer_with_symbols(monkeypatch, symbols):
    an = MarketAnalyzer.__new__(MarketAnalyzer)  # skip __init__ (no I/O)
    an.excluded = set()
    an._symbols_ts = None
    an.symbols = list(symbols)
    an.last_run_aborted = False
    return an


def _stub_post_burst(monkeypatch):
    """Neutralize the post-burst pipeline (filter/db/bottom/boost)."""
    class _StubScorer:
        @staticmethod
        def filter_signals(results, **k):
            return results

    class _StubDB:
        def log_run(self, **k):
            return 1

        def log_recommendation(self, run_id, rec):
            return None

    monkeypatch.setattr(anmod, "scorer", _StubScorer(), raising=False)
    monkeypatch.setattr(anmod, "db", _StubDB(), raising=False)
    bsmod = importlib.import_module("src.analysis.bottom_scanner")
    monkeypatch.setattr(bsmod.bottom_scanner, "scan",
                        lambda *a, **k: [], raising=False)
    monkeypatch.setattr(cfgmod.settings, "BOTTOM_BOOST_ENABLED", False,
                        raising=False)


def make_manager(tmp_path, monkeypatch):
    """RiskManager isolated from real data files."""
    pos_file = tmp_path / "open_positions.json"
    stats_file = tmp_path / "daily_stats.json"
    pending_file = tmp_path / "pending_entries.json"
    pos_file.write_text("[]")
    stats_file.write_text("{}")
    pending_file.write_text("[]")
    monkeypatch.setattr(manager_module, "POSITIONS_FILE", pos_file)
    monkeypatch.setattr(manager_module, "DAILY_STATS_FILE", stats_file)
    monkeypatch.setattr(manager_module, "PENDING_FILE", pending_file)

    class _NoDB:
        def __getattr__(self, name):
            return lambda *a, **k: None

    monkeypatch.setattr(manager_module, "db", _NoDB())
    return RiskManager(capital=10000)


# ============================================
# 1. fibonacci: entry-anchored geometry
# ============================================

def _fib_up():
    """Up impulse 100 -> 130; golden pocket 111.1-115.0 (0.618/0.5)."""
    return {
        "impulse": "up",
        "swing_high": 130.0,
        "swing_low": 100.0,
        "range": 30.0,
        "retracements": {"0.236": 122.9, "0.382": 118.9, "0.5": 115.0,
                         "0.618": 111.1, "0.786": 106.1},
        "extensions": {"1.272": 138.2, "1.414": 142.4, "1.618": 148.5},
        "golden_zone": {"low": 111.1, "high": 115.0},
    }


def _fib_down():
    """Down impulse 130 -> 100; pullback pocket 115.0-118.6."""
    return {
        "impulse": "down",
        "swing_high": 130.0,
        "swing_low": 100.0,
        "range": 30.0,
        "retracements": {"0.236": 107.1, "0.382": 111.5, "0.5": 115.0,
                         "0.618": 118.6, "0.786": 123.6},
        "extensions": {"1.272": 91.8, "1.414": 87.6, "1.618": 81.5},
        "golden_zone": {"low": 115.0, "high": 118.6},
    }


def test_bullish_limit_sl_anchored_to_entry():
    """The SOLUSDT bug: price 124, pocket 111.1-115, 10-bar swing low 119.8.
    The old SL (current-based) landed ABOVE the pocket fill -> instant stop.
    The new SL must sit BELOW the limit entry."""
    out = compute_entry_exit(
        direction="bullish", current_price=124.0, atr_val=2.0,
        sr={"supports": [], "resistances": []}, fib=_fib_up(),
        swing_low=119.82, swing_high=130.0, min_rr=1.5)
    assert out["entry_type"] == "limit"
    assert out["entry_price"] == pytest.approx(111.1)
    assert out["stop_loss"] < out["entry_price"], (
        "SL must be below the price a limit fill actually pays")
    assert out["stop_loss"] == pytest.approx(111.1 - 0.9 * 2.0)
    assert out["take_profit"] > out["entry_price"]
    assert out["risk_reward_ratio"] > 0


def test_bullish_limit_rr_measured_from_entry():
    """RR/distances are relative to the entry, not the analysis-time price."""
    out = compute_entry_exit(
        direction="bullish", current_price=124.0, atr_val=2.0,
        sr={"supports": [], "resistances": []}, fib=_fib_up(),
        swing_low=119.82, swing_high=130.0, min_rr=1.5)
    entry, sl, tp = out["entry_price"], out["stop_loss"], out["take_profit"]
    assert out["risk_reward_ratio"] == pytest.approx((tp - entry) / (entry - sl))
    assert out["sl_distance_pct"] == pytest.approx((entry - sl) / entry * 100,
                                                   rel=1e-6)


def test_bullish_market_entry_geometry_unchanged():
    """Price already inside the pocket -> market entry; geometry identical
    to the pre-v5.17 behavior (entry == current price)."""
    out = compute_entry_exit(
        direction="bullish", current_price=113.0, atr_val=2.0,
        sr={"supports": [], "resistances": []}, fib=_fib_up(),
        swing_low=119.82, swing_high=130.0, min_rr=1.5)
    assert out["entry_type"] == "market"
    assert out["entry_price"] == pytest.approx(113.0)
    # struct_dist = (113 - 119.82) + 0.8 < 0 -> clamped to 0.9 x ATR
    assert out["stop_loss"] == pytest.approx(113.0 - 0.9 * 2.0)
    assert out["stop_loss"] < out["entry_price"]


def test_bearish_limit_sl_anchored_above_entry():
    """Mirror bug for shorts: a limit short fills at the pocket ABOVE the
    price seen at analysis time; the stop must sit above the FILL."""
    out = compute_entry_exit(
        direction="bearish", current_price=90.0, atr_val=2.0,
        sr={"supports": [], "resistances": []}, fib=_fib_down(),
        swing_low=95.0, swing_high=92.0, min_rr=1.5)
    assert out["entry_type"] == "limit"
    assert out["entry_price"] == pytest.approx(118.6)
    assert out["stop_loss"] > out["entry_price"], (
        "short SL must be above the price the short actually sells at")
    assert out["take_profit"] < out["entry_price"]
    assert out["risk_reward_ratio"] > 0


def test_targets_filtered_against_entry_not_current():
    """A resistance between the pocket and the current price is a valid TP1
    for a pocket fill (it was invisible to the old current-price filter)."""
    out = compute_entry_exit(
        direction="bullish", current_price=124.0, atr_val=2.0,
        sr={"supports": [], "resistances": [120.0]}, fib=_fib_up(),
        swing_low=119.82, swing_high=130.0, min_rr=1.5)
    # 120 > entry 111.1 -> eligible; closest-first order wins over 130
    assert out["take_profit"] == pytest.approx(120.0)


# ============================================
# 2. validate_recommendation: coherence catch-all
# ============================================

def _valid_rec(**over):
    rec = {
        "symbol": "XUSDT", "direction": "bullish",
        "confidence": 80.0, "admission_confidence": 80.0,
        "risk_reward_ratio": 2.5, "expected_rise_pct": 3.0,
        "stop_loss": 95.0, "current_price": 100.0,
        "harmony": 0.9, "decision": {},
    }
    rec.update(over)
    return rec


@pytest.fixture
def _static_gates(monkeypatch):
    monkeypatch.setattr(cfgmod.settings, "REGIME_ENABLED", False,
                        raising=False)


def test_validate_rejects_long_with_sl_above_entry(_static_gates):
    mgr = RiskManager.__new__(RiskManager)
    ok, reasons = mgr.validate_recommendation(
        _valid_rec(stop_loss=101.0, current_price=100.0))
    assert ok is False
    assert any("Incoherent stop" in r for r in reasons)


def test_validate_rejects_short_with_sl_below_entry(_static_gates):
    mgr = RiskManager.__new__(RiskManager)
    ok, reasons = mgr.validate_recommendation(
        _valid_rec(direction="bearish", stop_loss=99.0, current_price=100.0))
    assert ok is False
    assert any("Incoherent stop" in r for r in reasons)


def test_validate_accepts_coherent_rec(_static_gates):
    mgr = RiskManager.__new__(RiskManager)
    ok, reasons = mgr.validate_recommendation(_valid_rec())
    assert ok is True, reasons


def test_validate_ignores_neutral_direction(_static_gates):
    """Neutral recs (AI/display only) are not the coherence gate's target."""
    mgr = RiskManager.__new__(RiskManager)
    ok, reasons = mgr.validate_recommendation(
        _valid_rec(direction="neutral", stop_loss=101.0))
    assert ok is True, reasons


# ============================================
# 3. cycle: momentum bypass hard cap
# ============================================

def _cycle_rec(price, conf=99.0, a_plus=True):
    return {
        "symbol": "AAAUSDT", "direction": "bullish",
        "entry_type": "limit", "entry_zone": {"low": 100.0, "high": 102.0},
        "current_price": price, "entry_price": 100.0,
        "stop_loss": 99.0, "take_profit": 110.0, "take_profit_2": 115.0,
        "atr": 2.0, "confidence": conf, "admission_confidence": conf,
        "expected_rise_pct": 3.0, "risk_reward_ratio": 2.0,
        "harmony": 0.9, "a_plus": a_plus, "decision": {},
    }


def _stub_cycle_gates(monkeypatch):
    monkeypatch.setattr(cfgmod.settings, "REGIME_ENABLED", False,
                        raising=False)
    monkeypatch.setattr("src.risk.manager.risk_manager.can_open_position",
                        lambda *a, **k: True, raising=False)
    monkeypatch.setattr("src.risk.manager.risk_manager.is_symbol_blocked",
                        lambda *a, **k: False, raising=False)
    monkeypatch.setattr("src.risk.manager.risk_manager.market_tide_blocked",
                        lambda *a, **k: (False, ""), raising=False)
    monkeypatch.setattr("src.risk.manager.risk_manager.has_open_position",
                        lambda *a, **k: False, raising=False)
    monkeypatch.setattr("src.risk.manager.risk_manager.open_positions",
                        [], raising=False)
    monkeypatch.setattr("src.risk.manager.risk_manager.pending_entries",
                        [], raising=False)
    monkeypatch.setattr(cycmod.db, "get_daily_stats", lambda *a, **k: [],
                        raising=False)
    monkeypatch.setattr(cycmod.telegram_notifier, "enabled", False,
                        raising=False)


def test_far_above_zone_pends_even_for_a_plus(monkeypatch):
    """4 ATR above the zone: even an A+ 99% setup must NOT chase."""
    _stub_cycle_gates(monkeypatch)
    pending_calls, opened_calls = [], []

    def fake_arm(rec, reason=""):
        pending_calls.append(rec["symbol"])
        return {"zone_low": 100.0, "zone_high": 102.0}

    def fake_open(rec):
        opened_calls.append(rec["symbol"])
        return {"status": "opened"}

    monkeypatch.setattr("src.risk.manager.risk_manager.add_pending_entry",
                        fake_arm, raising=False)
    monkeypatch.setattr("src.risk.manager.risk_manager.open_position",
                        fake_open, raising=False)
    from src.core.cycle import open_new_positions
    open_new_positions([_cycle_rec(price=110.0)])  # 4 ATR above zone_high
    assert pending_calls == ["AAAUSDT"]
    assert opened_calls == []


def test_far_above_zone_pends_for_high_confidence(monkeypatch):
    """admission_confidence 95 without a_plus: the OLD behavior opened here
    (AVAX 90.9 chased +18%). Now it waits for the pullback."""
    _stub_cycle_gates(monkeypatch)
    pending_calls, opened_calls = [], []

    def fake_arm(rec, reason=""):
        pending_calls.append(rec["symbol"])
        return {"zone_low": 100.0, "zone_high": 102.0}

    def fake_open(rec):
        opened_calls.append(rec["symbol"])
        return {"status": "opened"}

    monkeypatch.setattr("src.risk.manager.risk_manager.add_pending_entry",
                        fake_arm, raising=False)
    monkeypatch.setattr("src.risk.manager.risk_manager.open_position",
                        fake_open, raising=False)
    from src.core.cycle import open_new_positions
    open_new_positions([_cycle_rec(price=110.0, conf=95.0, a_plus=False)])
    assert pending_calls == ["AAAUSDT"]
    assert opened_calls == []


def test_near_zone_momentum_still_opens(monkeypatch):
    """0.5 ATR above the zone with momentum: the bypass keeps working."""
    _stub_cycle_gates(monkeypatch)
    pending_calls, opened_calls = [], []

    def fake_arm(rec, reason=""):
        pending_calls.append(rec["symbol"])
        return {"zone_low": 100.0, "zone_high": 102.0}

    def fake_open(rec):
        opened_calls.append(rec["symbol"])
        return {"status": "opened"}

    monkeypatch.setattr("src.risk.manager.risk_manager.add_pending_entry",
                        fake_arm, raising=False)
    monkeypatch.setattr("src.risk.manager.risk_manager.open_position",
                        fake_open, raising=False)
    from src.core.cycle import open_new_positions
    open_new_positions([_cycle_rec(price=103.0)])  # 0.5 ATR above zone_high
    assert opened_calls == ["AAAUSDT"]
    assert pending_calls == []


def test_setting_pending_momentum_max_atr_default():
    assert cfgmod.settings.PENDING_MOMENTUM_MAX_ATR == pytest.approx(1.0)
    assert cfgmod.settings.PENDING_MOMENTUM_MAX_ATR > cfgmod.settings.PENDING_CHASE_ATR


# ============================================
# 4. analyzer: ws_only bursts survive a REST ban
# ============================================

def test_ws_only_burst_survives_ban(monkeypatch, isolated_rec_file):
    """The production bug: cooldown active + ws_only cycle -> the old
    analyze_one skipped every symbol and the burst aborted 80/80. With the
    fix the ws_only burst runs and the last-good snapshot is replaced by a
    fresh WS-cache analysis."""
    limiter = make_limiter()
    monkeypatch.setattr("src.core.rate_limiter.rate_limiter", limiter,
                        raising=False)
    limiter.trigger_cooldown(598.0)
    assert _ban_active()

    an = _analyzer_with_symbols(monkeypatch, ["AAAUSDT", "BBBUSDT"])

    def fake_analyze_one(self, symbol, ws_only=False):
        if _ban_active() and not ws_only:
            return {"symbol": symbol, "skip": True, "reason": "rate ban"}
        return {"symbol": symbol, "skip": False, "confidence": 90,
                "expected_rise_pct": 3.0}

    monkeypatch.setattr(MarketAnalyzer, "analyze_one", fake_analyze_one)
    _stub_post_burst(monkeypatch)

    out = an.analyze_all(parallel=False, ws_only=True)
    assert an.last_run_aborted is False
    assert [r["symbol"] for r in out] == ["AAAUSDT", "BBBUSDT"]


def test_rest_burst_still_aborts_during_ban(monkeypatch, isolated_rec_file):
    """Regression guard: REST-capable bursts must keep the v5.12 abort
    behavior (fail fast, keep the last good snapshot)."""
    limiter = make_limiter()
    monkeypatch.setattr("src.core.rate_limiter.rate_limiter", limiter,
                        raising=False)
    limiter.trigger_cooldown(598.0)

    an = _analyzer_with_symbols(monkeypatch, [f"S{i}USDT" for i in range(20)])

    def fake_analyze_one(self, symbol, ws_only=False):
        if _ban_active() and not ws_only:
            return {"symbol": symbol, "skip": True, "reason": "rate ban"}
        return {"symbol": symbol, "skip": False, "confidence": 90,
                "expected_rise_pct": 3.0}

    monkeypatch.setattr(MarketAnalyzer, "analyze_one", fake_analyze_one)
    _stub_post_burst(monkeypatch)

    out = an.analyze_all(parallel=True, max_workers=4)
    assert out == []
    assert an.last_run_aborted is True
    snap = json.loads(isolated_rec_file.read_text(encoding="utf-8"))
    assert snap["top_recommendations"] == [{"symbol": "OLD"}]


def test_analyze_one_ws_only_runs_during_ban(monkeypatch):
    """The real analyze_one: ws_only during a ban must proceed (candles from
    the WS cache, order book skipped); the REST path still short-circuits."""
    limiter = make_limiter()
    monkeypatch.setattr("src.core.rate_limiter.rate_limiter", limiter,
                        raising=False)
    limiter.trigger_cooldown(598.0)
    an = _analyzer_with_symbols(monkeypatch, ["AAAUSDT"])

    def _df(n=80):
        return pd.DataFrame({
            "open": [1.0] * n, "high": [1.1] * n, "low": [0.9] * n,
            "close": [1.05] * n, "volume": [1000.0] * n,
        })

    class _StubDF:
        @staticmethod
        def get_multi_timeframe_candles(symbol, intervals, limit=None):
            return {tf: _df() for tf in intervals}

        @staticmethod
        def get_order_book(*a, **k):
            raise AssertionError("order book must not be fetched in ws_only")

    class _StubScorer:
        @staticmethod
        def analyze_symbol(df, symbol, **k):
            return {"symbol": symbol, "skip": False, "confidence": 88.0,
                    "expected_rise_pct": 3.0}

    monkeypatch.setattr(anmod, "data_fetcher", _StubDF(), raising=False)
    monkeypatch.setattr(anmod, "scorer", _StubScorer(), raising=False)

    r = an.analyze_one("AAAUSDT", ws_only=True)
    assert not r.get("skip"), r
    assert r["confidence"] == 88.0

    r2 = an.analyze_one("AAAUSDT", ws_only=False)
    assert r2.get("skip") is True
    assert r2.get("reason") == "rate ban"


# ============================================
# 5. pending fills: coherent geometry end-to-end
# ============================================

def _pending_rec(price, sl):
    return {
        "symbol": "TESTUSDT", "direction": "bullish",
        "entry_type": "limit", "entry_zone": {"low": 100.0, "high": 102.0},
        "current_price": price, "entry_price": 100.0,
        "stop_loss": sl, "take_profit": 112.0, "take_profit_2": 120.0,
        "atr": 2.0, "harmony": 0.9, "confidence": 75,
        "expected_rise_pct": 3.0, "risk_reward_ratio": 2.0, "decision": {},
    }


def test_pending_fill_rejected_when_sl_above_fill(tmp_path, monkeypatch):
    """A stale rec whose SL sits above the zone (the SOLUSDT shape) must be
    rejected AT FILL TIME by the coherence gate, not opened and instantly
    stopped out."""
    rm = make_manager(tmp_path, monkeypatch)
    rm.add_pending_entry(_pending_rec(price=110.0, sl=105.0), "test")
    filled = rm.check_pending_fills({"TESTUSDT": 101.5})
    assert filled == []
    assert rm.open_positions == []
    assert rm.pending_entries == []  # rejected fill -> dropped, no retry


def test_pending_fill_with_anchored_sl_opens(tmp_path, monkeypatch):
    """Pocket-anchored SL (new geometry) -> the fill opens a coherent long."""
    rm = make_manager(tmp_path, monkeypatch)
    rm.add_pending_entry(_pending_rec(price=110.0, sl=99.0), "test")
    filled = rm.check_pending_fills({"TESTUSDT": 101.5})
    assert len(filled) == 1
    pos = rm.open_positions[0]
    assert pos["entry_price"] == pytest.approx(101.5)
    assert pos["stop_loss"] == pytest.approx(99.0)
    assert pos["stop_loss"] < pos["entry_price"]
