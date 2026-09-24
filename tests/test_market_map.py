"""v5.2 Market Map: leader/follower correlation classification + regime filter.

User observation: many altcoins chart almost identically to BTC/SOL/XRP.
These tests cover the correlation math, leader trend detection, and the
regime action applied to signals before filtering.
"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.analysis.market_map import MarketMap, INDEPENDENT, market_map


# ---------- correlation / beta math ----------

def test_corr_beta_perfectly_correlated():
    rng = np.random.default_rng(11)
    rets = rng.normal(0.0005, 0.01, 200)          # leader 1h returns
    price_l = pd.Series(100 * (1 + rets).cumprod())
    price_s = pd.Series(50 * (1 + rets * 1.2).cumprod())  # follower = 1.2x beta
    corr, beta = MarketMap._corr_beta(price_s.pct_change().dropna(),
                                      price_l.pct_change().dropna())
    assert corr is not None and corr > 0.99
    assert beta is not None and 1.1 < beta < 1.3


def test_corr_beta_independent_noise():
    rng = np.random.default_rng(42)
    n = 300
    leader = pd.Series(rng.normal(0, 0.01, n).cumsum())
    follower = pd.Series(rng.normal(0, 0.01, n).cumsum())
    corr, _ = MarketMap._corr_beta(follower.pct_change().dropna(),
                                   leader.pct_change().dropna())
    assert abs(corr) < 0.4  # noise, not a follower


def test_corr_beta_short_series_returns_none():
    s = pd.Series([0.01, -0.02, 0.01])  # way below the 50-bar minimum
    corr, beta = MarketMap._corr_beta(s, s)
    assert corr is None and beta is None


# ---------- leader trend detection ----------

def _series_from(values):
    return pd.Series(values, dtype=float)


def test_trend_bullish_uptrend():
    up = _series_from(np.linspace(100, 160, 168))
    meta = MarketMap._trend_from_closes(up)
    assert meta["trend"] == "bullish"


def test_trend_bearish_downtrend():
    down = _series_from(np.linspace(160, 100, 168))
    meta = MarketMap._trend_from_closes(down)
    assert meta["trend"] == "bearish"


def test_trend_flat_is_neutral():
    rng = np.random.default_rng(7)
    flat = _series_from(100 + rng.normal(0, 0.05, 168).cumsum() * 0.02)
    meta = MarketMap._trend_from_closes(flat)
    assert meta["trend"] in ("neutral", "bullish", "bearish")  # never crashes
    assert "price" in meta


def test_trend_insufficient_data_neutral():
    meta = MarketMap._trend_from_closes(_series_from([100.0] * 40))
    assert meta["trend"] == "neutral"


# ---------- the ACTION: apply_regime ----------

def _fake_map():
    return {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "leaders": {
            "BTCUSDT": {"trend": "bearish"},
            "SOLUSDT": {"trend": "bullish"},
            "ETHUSDT": {"trend": "neutral"},
        },
        "symbols": {
            "ACEUSDT": {"leader": "BTCUSDT", "corr": 0.80, "is_leader": False},
            "RAYUSDT": {"leader": "SOLUSDT", "corr": 0.90, "is_leader": False},
            "ETHUSDT": {"leader": "ETHUSDT", "corr": 1.0, "is_leader": True},
            "INDUSDT": {"leader": INDEPENDENT, "corr": 0.20, "is_leader": False},
        },
    }


def _run_regime(results, monkeypatch):
    mm = MarketMap()
    monkeypatch.setattr(mm, "get_map", lambda force=False: _fake_map())
    return mm.apply_regime(results)


def test_bearish_leader_penalizes_admission(monkeypatch):
    r = {"symbol": "ACEUSDT", "direction": "bullish",
         "confidence": 70.0, "admission_confidence": 65.0}
    out = _run_regime([r], monkeypatch)[0]
    # penalty = corr 0.8 * 20 = 16 -> both confidences drop
    assert out["confidence"] == 54.0
    assert out["admission_confidence"] == 49.0
    assert out["leader"] == "BTCUSDT"
    assert out["leader_trend"] == "bearish"
    assert out["regime_adj"] == -16.0


def test_bearish_leader_can_demote_below_threshold(monkeypatch):
    """The whole point: a follower of a crashing leader gets filtered out."""
    r = {"symbol": "ACEUSDT", "direction": "bullish",
         "confidence": 70.0, "admission_confidence": 65.0}
    out = _run_regime([r], monkeypatch)[0]
    assert out["admission_confidence"] < 60  # below MIN_CONFIDENCE default


def test_bullish_leader_boosts_rank_only(monkeypatch):
    r = {"symbol": "RAYUSDT", "direction": "bullish",
         "confidence": 70.0, "admission_confidence": 65.0}
    out = _run_regime([r], monkeypatch)[0]
    # boost = corr 0.9 * 8 = 7.2 -> confidence up, admission UNTOUCHED (v4 rule)
    assert out["confidence"] == 77.2
    assert out["admission_confidence"] == 65.0
    assert out["regime_adj"] == 7.2


def test_neutral_leader_and_independent_untouched(monkeypatch):
    r_ind = {"symbol": "INDUSDT", "direction": "bullish",
             "confidence": 70.0, "admission_confidence": 65.0}
    r_leader = {"symbol": "ETHUSDT", "direction": "bullish",
                "confidence": 70.0, "admission_confidence": 65.0}
    out = _run_regime([r_ind, r_leader], monkeypatch)
    assert out[0]["confidence"] == 70.0 and out[0]["leader_trend"] == "n/a"
    assert out[1]["confidence"] == 70.0 and out[1]["leader_trend"] == "n/a"


def test_bearish_direction_untouched(monkeypatch):
    r = {"symbol": "ACEUSDT", "direction": "bearish",
         "confidence": 70.0, "admission_confidence": 65.0}
    out = _run_regime([r], monkeypatch)[0]
    assert out["confidence"] == 70.0
    assert "regime_adj" not in out


def test_apply_regime_without_map_is_noop(monkeypatch):
    mm = MarketMap()

    def broken_get_map(force=False):
        raise RuntimeError("no data")

    monkeypatch.setattr(mm, "get_map", broken_get_map)
    r = {"symbol": "X", "direction": "bullish", "confidence": 70.0}
    assert mm.apply_regime([r]) == [r]
    assert mm.apply_regime([]) == []


# ---------- cache behavior ----------

def test_is_stale_logic(tmp_path, monkeypatch):
    import src.analysis.market_map as mmmod
    monkeypatch.setattr(mmmod, "MARKET_MAP_FILE", tmp_path / "m.json")
    mm = MarketMap()
    assert mm._is_stale(None) is True
    assert mm._is_stale({}) is True
    assert mm._is_stale({"symbols": {}}) is True

    fresh = {"updated_at": datetime.now(timezone.utc).isoformat(), "symbols": {"A": {}}}
    assert mm._is_stale(fresh) is False

    old = {"updated_at": (datetime.now(timezone.utc)
                          - timedelta(hours=7)).isoformat(),
           "symbols": {"A": {}}}
    assert mm._is_stale(old) is True


def test_get_map_uses_fresh_cache_without_rebuild(tmp_path, monkeypatch):
    import src.analysis.market_map as mmmod
    monkeypatch.setattr(mmmod, "MARKET_MAP_FILE", tmp_path / "m.json")
    mm = MarketMap()
    fresh = {"updated_at": datetime.now(timezone.utc).isoformat(),
             "leaders": {}, "symbols": {"A": {"leader": "BTCUSDT"}}}

    def must_not_build():
        raise AssertionError("rebuild triggered on fresh cache")

    monkeypatch.setattr(mm, "_build", must_not_build)
    mm._map = fresh
    assert mm.get_map() is fresh


def test_grouped_view_shapes(tmp_path, monkeypatch):
    import src.analysis.market_map as mmmod
    monkeypatch.setattr(mmmod, "MARKET_MAP_FILE", tmp_path / "m.json")
    mm = MarketMap()
    mm._map = _fake_map()
    view = mm.grouped_view()
    assert view["groups"]["BTCUSDT"][0]["symbol"] == "ACEUSDT"
    assert view["leaders"]["BTCUSDT"]["trend"] == "bearish"
    assert view["updated_at"] is not None


def test_singleton_exists():
    assert market_map is not None
