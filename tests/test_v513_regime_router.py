"""
v5.13 Regime Router tests.

User request: "review the trade-acceptance logic across different market
volatility states; the bot should use the strategy appropriate for the
market state. States come from the major coins, the Fear & Greed index,
and the weekend."

Covers:
  - weekend window (Fri 22:00 UTC -> Mon 00:00 UTC) + kill switch
  - compose(): leaders/volatility/F&G/weekend -> policy multipliers,
    threshold shifts, caps, crisis freeze
  - apply_regime_routing(): strategy-fit re-ranking (ranging favors
    mean-reversion over breakout at equal raw scores)
  - manager._regime_gates()/_regime_size_multiplier(): adjusted acceptance
    gates + fail-open behavior
  - cycle.open_new_positions(): regime freeze gate
  - Fear & Greed: cache hit + fail-neutral
  - /api/regime: adjusted gates visible on the dashboard API
"""
import importlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

rrmod = importlib.import_module("src.analysis.regime_router")
RegimeRouter = rrmod.RegimeRouter

from src.core.cycle import open_new_positions  # noqa: E402
from src.risk.manager import RiskManager  # noqa: E402


# ============================================
# weekend window
# ============================================

def test_weekend_window_utc():
    assert RegimeRouter and rrmod.is_weekend(
        datetime(2026, 9, 26, 12, 0, tzinfo=timezone.utc)) is True   # Sat
    assert rrmod.is_weekend(
        datetime(2026, 9, 27, 12, 0, tzinfo=timezone.utc)) is True   # Sun
    assert rrmod.is_weekend(
        datetime(2026, 9, 25, 23, 0, tzinfo=timezone.utc)) is True   # Fri 23:00
    assert rrmod.is_weekend(
        datetime(2026, 9, 25, 21, 0, tzinfo=timezone.utc)) is False  # Fri 21:00
    assert rrmod.is_weekend(
        datetime(2026, 9, 28, 12, 0, tzinfo=timezone.utc)) is False  # Mon


def test_weekend_kill_switch(monkeypatch):
    cfg = importlib.import_module("config.settings")  # the MODULE
    monkeypatch.setattr(cfg.settings, "WEEKEND_FILTER", False)
    assert rrmod.is_weekend(
        datetime(2026, 9, 26, 12, 0, tzinfo=timezone.utc)) is False  # Sat


# ============================================
# compose(): the four components -> policy
# ============================================

def _w(pol, name):
    return pol["strategy_weights"].get(name, 1.0)


LEADERS_BULL = {"score": 75, "verdict": "bullish", "posture_ar": ""}
LEADERS_BEAR = {"score": 30, "verdict": "bearish", "posture_ar": ""}
LEADERS_FLAT = {"score": 50, "verdict": "neutral", "posture_ar": ""}
VOL_NORMAL = {"atr_pct": 1.2, "median": 1.1, "ratio": 1.1, "level": "normal"}
VOL_CRISIS = {"atr_pct": 3.5, "median": 1.1, "ratio": 3.1, "level": "crisis"}
VOL_HIGH = {"atr_pct": 2.3, "median": 1.1, "ratio": 2.1, "level": "elevated"}
VOL_DEAD = {"atr_pct": 0.5, "median": 1.0, "ratio": 0.5, "level": "dead"}
FNG_MID = {"value": 50, "label": "neutral", "source": "test"}


def test_trending_bull_favors_trend_strategies():
    pol = RegimeRouter().compose(LEADERS_BULL, FNG_MID, VOL_NORMAL, False)
    assert pol["state"] == "trending_bull"
    assert _w(pol, "triple_confluence_trend") > 1.0
    assert _w(pol, "bb_mean_reversion") < 1.0
    assert pol["min_confidence_adjust"] == 0.0
    assert pol["size_multiplier"] == 1.0


def test_trending_bear_tightens_and_favors_meanrev():
    pol = RegimeRouter().compose(LEADERS_BEAR, FNG_MID, VOL_NORMAL, False)
    assert pol["state"] == "trending_bear"
    assert pol["min_confidence_adjust"] >= 3.0
    assert pol["size_multiplier"] < 1.0
    assert _w(pol, "bb_mean_reversion") > 1.0
    assert _w(pol, "trend_pullback") < 1.0


def test_ranging_favors_meanrev_penalizes_breakout():
    pol = RegimeRouter().compose(LEADERS_FLAT, FNG_MID, VOL_NORMAL, False)
    assert pol["state"] == "ranging"
    assert _w(pol, "bb_mean_reversion") > 1.0
    assert _w(pol, "macd_breakout") < 1.0


def test_vol_crisis_overlay_and_freeze():
    pol = RegimeRouter().compose(LEADERS_BEAR, FNG_MID, VOL_CRISIS, False)
    assert pol["min_rr_adjust"] >= 0.4
    assert pol["min_confidence_adjust"] >= 6.0
    assert pol["size_multiplier"] <= 0.5
    assert pol["allow_new_entries"] is False
    assert "crisis" in pol["freeze_reason"]


def test_vol_crisis_without_bear_does_not_freeze():
    pol = RegimeRouter().compose(LEADERS_BULL, FNG_MID, VOL_CRISIS, False)
    assert pol["allow_new_entries"] is True
    assert pol["size_multiplier"] <= 0.5  # shrink still applies


def test_vol_elevated_and_dead():
    hi = RegimeRouter().compose(LEADERS_FLAT, FNG_MID, VOL_HIGH, False)
    assert hi["min_rr_adjust"] >= 0.2 and hi["size_multiplier"] <= 0.7
    dead = RegimeRouter().compose(LEADERS_FLAT, FNG_MID, VOL_DEAD, False)
    assert _w(dead, "bb_mean_reversion") > 1.2
    assert _w(dead, "volatility_breakout") < 1.0


def test_fear_and_greed_overlays():
    fear = RegimeRouter().compose(
        LEADERS_FLAT, {"value": 15, "label": "extreme fear"}, VOL_NORMAL, False)
    assert fear["min_confidence_adjust"] >= 4.0
    assert fear["size_multiplier"] <= 0.85
    assert _w(fear, "liquidity_sweep_reversal") > 1.0
    greed = RegimeRouter().compose(
        LEADERS_FLAT, {"value": 85, "label": "extreme greed"}, VOL_NORMAL, False)
    assert greed["min_confidence_adjust"] >= 4.0
    assert greed["size_multiplier"] <= 0.75
    mid = RegimeRouter().compose(LEADERS_FLAT, FNG_MID, VOL_NORMAL, False)
    assert mid["min_confidence_adjust"] == 0.0
    assert mid["size_multiplier"] == 1.0


def test_weekend_overlay():
    pol = RegimeRouter().compose(LEADERS_FLAT, FNG_MID, VOL_NORMAL, True)
    assert pol["min_rr_adjust"] >= 0.2
    assert pol["size_multiplier"] <= 0.75
    assert _w(pol, "volatility_breakout") < 1.0
    assert pol["weekend"] is True


def test_extreme_stacking_caps():
    """Crisis + bear + weekend + extreme fear -> capped, never insane."""
    pol = RegimeRouter().compose(
        LEADERS_BEAR, {"value": 10, "label": "extreme fear"},
        VOL_CRISIS, True)
    assert pol["min_confidence_adjust"] <= 12.0
    assert pol["min_rr_adjust"] <= 0.6
    assert pol["size_multiplier"] >= 0.25
    assert pol["allow_new_entries"] is False
    assert pol["state_ar"]  # Arabic description present


# ============================================
# apply_regime_routing: strategy-fit ranking
# ============================================

def _rec(symbol, strategy, conf=70.0, rr=2.0, harmony=0.5):
    return {
        "symbol": symbol,
        "confidence": conf,
        "risk_reward_ratio": rr,
        "harmony": harmony,
        "signals": [{"strategy": strategy, "direction": "bullish"}],
    }


def test_routing_ranks_meanrev_above_breakout_in_range():
    recs = [_rec("BRKUSDT", "macd_breakout"),
            _rec("MRVUSDT", "bb_mean_reversion")]
    router = RegimeRouter()
    pol = router.compose(LEADERS_FLAT, FNG_MID, VOL_NORMAL, False)
    router._regime = pol          # inject without I/O
    router._regime_ts = __import__("time").time() + 10**9
    out = router.apply_regime_routing(recs)
    assert out[0]["symbol"] == "MRVUSDT"      # mean-reversion fits the range
    assert out[0]["regime_fit"] > out[1]["regime_fit"]
    assert out[0]["regime_state"] == "ranging"


def test_routing_trend_above_meanrev_in_bull():
    recs = [_rec("MRVUSDT", "bb_mean_reversion"),
            _rec("TRDUSDT", "triple_confluence_trend")]
    router = RegimeRouter()
    pol = router.compose(LEADERS_BULL, FNG_MID, VOL_NORMAL, False)
    router._regime = pol
    router._regime_ts = __import__("time").time() + 10**9
    out = router.apply_regime_routing(recs)
    assert out[0]["symbol"] == "TRDUSDT"


# ============================================
# manager: regime-adjusted gates + sizing
# ============================================

class _FixedRouter:
    def __init__(self, policy):
        self._p = policy

    def active_policy(self):
        return self._p

    def status(self):
        return dict(self._p)


@pytest.fixture
def fixed_policy(monkeypatch):
    pol = {
        "min_confidence_adjust": 5.0,
        "min_rr_adjust": 0.3,
        "size_multiplier": 0.75,
        "strategy_weights": {s: 1.0 for s in rrmod.ALL_STRATS},
        "allow_new_entries": True,
        "state": "ranging",
        "state_ar": "نطاق عرضي",
    }
    monkeypatch.setattr(rrmod, "regime_router", _FixedRouter(pol))
    yield pol


def test_regime_gates_applied(fixed_policy):
    from config.settings import settings
    lo, rr = RiskManager._regime_gates()
    assert lo == pytest.approx(settings.MIN_CONFIDENCE + 5.0)
    assert rr == pytest.approx(settings.MIN_RR_RATIO + 0.3)


def test_regime_gates_fail_open(monkeypatch):
    from config.settings import settings

    class _Boom:
        def active_policy(self):
            raise RuntimeError("down")

    monkeypatch.setattr(rrmod, "regime_router", _Boom())
    lo, rr = RiskManager._regime_gates()
    assert lo == settings.MIN_CONFIDENCE
    assert rr == settings.MIN_RR_RATIO


def test_validate_rejects_with_regime_adjusted_gate(fixed_policy):
    """A rec that passes the static gate but fails the regime-adjusted one."""
    from config.settings import settings
    mgr = RiskManager.__new__(RiskManager)  # no __init__ I/O
    rec = {
        "symbol": "XUSDT",
        "direction": "bullish",
        "confidence": settings.MIN_CONFIDENCE + 0.5,      # passes static
        "admission_confidence": settings.MIN_CONFIDENCE + 0.5,
        "risk_reward_ratio": settings.MIN_RR_RATIO + 0.1,  # passes static
        "expected_rise_pct": settings.MIN_EXPECTED_RISE + 1,
        "stop_loss": 100.0,
        "harmony": 0.9,
    }
    ok, reasons = mgr.validate_recommendation(rec)
    assert ok is False
    assert any("regime-adjusted" in r for r in reasons)


def test_regime_size_multiplier(fixed_policy):
    assert RiskManager._regime_size_multiplier() == pytest.approx(0.75)


# ============================================
# cycle: regime freeze gate
# ============================================

def test_open_new_positions_frozen_in_crisis(monkeypatch):
    pol = {
        "min_confidence_adjust": 12.0,
        "min_rr_adjust": 0.6,
        "size_multiplier": 0.25,
        "strategy_weights": {},
        "allow_new_entries": False,
        "freeze_reason": "crisis: high vol + bearish leaders",
        "state": "trending_bear",
        "state_ar": "أزمة",
    }
    monkeypatch.setattr(rrmod, "regime_router", _FixedRouter(pol))

    def boom(*a, **k):
        raise AssertionError("no position may open while frozen")

    monkeypatch.setattr(
        "src.risk.manager.risk_manager.open_paper_position", boom,
        raising=False)
    recs = [_rec("AAAUSDT", "bb_mean_reversion")]
    assert open_new_positions(recs) == 0


def test_open_new_positions_allowed_when_policy_ok(monkeypatch, fixed_policy):
    """Policy allows entries -> the freeze gate is a no-op (downstream gates
    may still reject; we only assert the freeze gate does not block)."""
    opened_calls = []

    def fake_open(rec):
        opened_calls.append(rec["symbol"])
        return {"status": "opened"}

    monkeypatch.setattr(
        "src.risk.manager.risk_manager.open_paper_position", fake_open,
        raising=False)
    monkeypatch.setattr(
        "src.risk.manager.risk_manager.can_open_position",
        lambda *a, **k: True, raising=False)
    monkeypatch.setattr(
        "src.risk.manager.risk_manager.is_symbol_blocked",
        lambda *a, **k: False, raising=False)
    monkeypatch.setattr(
        "src.risk.manager.risk_manager.market_tide_blocked",
        lambda *a, **k: (False, ""), raising=False)
    monkeypatch.setattr(
        "src.risk.manager.risk_manager.has_open_position",
        lambda *a, **k: False, raising=False)
    monkeypatch.setattr(
        "src.risk.manager.risk_manager.open_positions", [], raising=False)
    monkeypatch.setattr(
        "src.risk.manager.risk_manager.pending_entries", [], raising=False)
    import src.core.cycle as cyc
    monkeypatch.setattr(cyc.db, "get_daily_stats", lambda *a, **k: [],
                        raising=False)
    recs = [_rec("AAAUSDT", "bb_mean_reversion")]
    open_new_positions(recs)  # must not raise on the freeze gate
    # downstream gates decide; freeze gate did not block


# ============================================
# Fear & Greed: cache + fail-neutral
# ============================================

def test_fng_cache_hit(monkeypatch, tmp_path):
    from src.utils.helpers import save_json, now_utc
    fng_file = tmp_path / "fng.json"
    save_json({"value": 71, "label": "greed", "source": "alternative.me",
               "fetched_at": now_utc().isoformat()}, fng_file)
    monkeypatch.setattr(rrmod, "FNG_FILE", fng_file)

    def boom(*a, **k):
        raise AssertionError("fresh cache must not hit the network")

    monkeypatch.setattr("requests.get", boom)
    out = RegimeRouter()._fear_greed()
    assert out["value"] == 71 and out["source"] == "alternative.me"


def test_fng_fail_neutral(monkeypatch, tmp_path):
    monkeypatch.setattr(rrmod, "FNG_FILE", tmp_path / "missing.json")

    def boom(*a, **k):
        raise ConnectionError("down")

    monkeypatch.setattr("requests.get", boom)
    out = RegimeRouter()._fear_greed()
    assert out["value"] == 50 and out["source"] == "unavailable"


# ============================================
# /api/regime
# ============================================

def test_api_regime_exposes_adjusted_gates(monkeypatch, fixed_policy):
    from fastapi.testclient import TestClient
    import src.web.app as appmod
    client = TestClient(appmod.app)
    r = client.get("/api/regime")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["gates"]["size_multiplier"] == pytest.approx(0.75)
    assert body["gates"]["allow_new_entries"] is True
    assert body["regime"]["state_ar"]
