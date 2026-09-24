"""v5.4: market map integrity + market cycle via leader coins + decision file.

Covers the root cause of the empty market-groups tab (a leaders-only /
holey map being cached as fresh for the whole 6h TTL), the leader-coin
market cycle (verdicts + weighted posture), and the Arabic classification
file (data/market_groups.txt).
"""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytest

import src.analysis.market_map as mmmod
from config.settings import settings
from src.analysis.market_map import MarketMap, INDEPENDENT


# ---------- helpers ----------

def _candles_df(base: float = 100.0, n: int = 200, drift: float = 0.0,
                seed: int = 3) -> pd.DataFrame:
    """OHLC 1h frame: per-bar drift + small deterministic noise."""
    rng = np.random.default_rng(seed)
    steps = drift + rng.normal(0, 0.001, n)
    close = base * np.cumprod(1 + steps)
    idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({
        "open": close * 0.999,
        "high": close * 1.004,
        "low": close * 0.996,
        "close": close,
        "volume": 1000.0,
    }, index=idx)


def _map_with_followers(fresh: bool = True) -> dict:
    ts = (datetime.now(timezone.utc) if fresh
          else datetime.now(timezone.utc) - timedelta(hours=7)).isoformat()
    return {
        "updated_at": ts,
        "corr_threshold": 0.55,
        "lookback_hours": 168,
        "universe_size": 5,
        "leaders": {
            "BTCUSDT": {"trend": "bullish"},
            "ETHUSDT": {"trend": "bearish"},
        },
        "symbols": {
            "BTCUSDT": {"leader": "BTCUSDT", "corr": 1.0, "is_leader": True},
            "ETHUSDT": {"leader": "ETHUSDT", "corr": 1.0, "is_leader": True},
            "F1USDT": {"leader": "BTCUSDT", "corr": 0.80, "beta": 1.1, "is_leader": False},
            "F2USDT": {"leader": "BTCUSDT", "corr": 0.60, "beta": 0.9, "is_leader": False},
            "F3USDT": {"leader": "ETHUSDT", "corr": 0.70, "beta": 1.0, "is_leader": False},
        },
    }


# ---------- v5.4 integrity: the empty-tab bug ----------

def test_leaders_only_map_is_stale(tmp_path, monkeypatch):
    """Regression: a map holding ONLY leader entries must be treated as
    stale - grouped_view() hides leaders, so this artifact kept the tab
    empty until the whole 6h TTL expired."""
    monkeypatch.setattr(mmmod, "MARKET_MAP_FILE", tmp_path / "m.json")
    mm = MarketMap()
    ts = datetime.now(timezone.utc).isoformat()
    leaders_only = {
        "updated_at": ts,
        "symbols": {
            "BTCUSDT": {"leader": "BTCUSDT", "is_leader": True},
            "ETHUSDT": {"leader": "ETHUSDT", "is_leader": True},
        },
    }
    assert mm._is_stale(leaders_only) is True

    with_followers = _map_with_followers(fresh=True)
    monkeypatch.setattr(mm, "_universe",
                        lambda: ["BTCUSDT", "ETHUSDT", "F1USDT", "F2USDT", "F3USDT"])
    assert mm._is_stale(with_followers) is False


def test_universe_growth_marks_stale(tmp_path, monkeypatch):
    """New symbols entering the universe must not stay unclassified for
    the whole TTL - a large growth forces a rebuild."""
    monkeypatch.setattr(mmmod, "MARKET_MAP_FILE", tmp_path / "m.json")
    mm = MarketMap()
    cache = _map_with_followers(fresh=True)  # universe_size = 5
    monkeypatch.setattr(mm, "_universe", lambda: [f"S{i}" for i in range(30)])
    assert mm._is_stale(cache) is True
    monkeypatch.setattr(mm, "_universe",
                        lambda: ["BTCUSDT", "ETHUSDT", "F1USDT", "F2USDT", "F3USDT"])
    assert mm._is_stale(cache) is False


def test_build_rejects_holey_map(tmp_path, monkeypatch):
    """50% of the universe failing to fetch candles must raise instead of
    caching a holey classification."""
    monkeypatch.setattr(mmmod, "MARKET_MAP_FILE", tmp_path / "m.json")
    monkeypatch.setattr(settings, "MARKET_MAP_MAX_FAIL_PCT", 0.25)
    mm = MarketMap()
    followers = [f"S{i}USDT" for i in range(10)]
    monkeypatch.setattr(mm, "_universe", lambda: ["BTCUSDT", "ETHUSDT"] + followers)

    def fake_candles(sym, interval="1h", limit=200):
        if sym in ("BTCUSDT", "ETHUSDT"):
            return _candles_df(drift=0.001)
        if sym in followers[:5]:  # 50% of followers fail (rate pressure)
            return None
        return _candles_df(drift=0.002, base=50.0, seed=5)

    monkeypatch.setattr(mmmod.data_fetcher, "get_candles", fake_candles)
    with pytest.raises(RuntimeError, match="Incomplete market map"):
        mm._build()
    # nothing holey was persisted
    assert not (tmp_path / "m.json").exists()


def test_build_records_integrity_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(mmmod, "MARKET_MAP_FILE", tmp_path / "m.json")
    mm = MarketMap()
    followers = [f"S{i}USDT" for i in range(5)]
    monkeypatch.setattr(mm, "_universe", lambda: ["BTCUSDT", "ETHUSDT"] + followers)

    def fake_candles(sym, interval="1h", limit=200):
        if sym in ("BTCUSDT", "ETHUSDT"):
            return _candles_df(drift=0.001)
        return _candles_df(drift=0.002, base=50.0, seed=7)

    monkeypatch.setattr(mmmod.data_fetcher, "get_candles", fake_candles)
    artifact = mm._build()
    assert artifact["universe_size"] == 7
    assert artifact["classified_count"] == 7
    assert artifact["failed_count"] == 0
    assert artifact["symbols"]["S0USDT"]["is_leader"] is False


def test_get_map_keeps_old_cache_when_build_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(mmmod, "MARKET_MAP_FILE", tmp_path / "m.json")
    (tmp_path / "m.json").write_text(
        json.dumps(_map_with_followers()), encoding="utf-8")
    mm = MarketMap()

    def broken_build():
        raise RuntimeError("Incomplete market map")

    monkeypatch.setattr(mm, "_build", broken_build)
    out = mm.get_map(force=True)  # force -> build attempted -> rejected
    assert out["symbols"]["F1USDT"]["leader"] == "BTCUSDT"  # old map served


def test_get_map_throttles_failed_rebuilds(tmp_path, monkeypatch):
    """Failed rebuilds retry at most once per 5 minutes (unless forced)."""
    monkeypatch.setattr(mmmod, "MARKET_MAP_FILE", tmp_path / "m.json")
    mm = MarketMap()
    calls = {"n": 0}

    def broken_build():
        calls["n"] += 1
        raise RuntimeError("boom")

    monkeypatch.setattr(mm, "_build", broken_build)
    out1 = mm.get_map()  # stale (no cache) -> attempt -> fail
    out2 = mm.get_map()  # throttled -> no second attempt
    assert calls["n"] == 1
    assert out1 == {"updated_at": None, "leaders": {}, "symbols": {}}
    assert out2 == out1
    mm.get_map(force=True)  # force bypasses the throttle
    assert calls["n"] == 2


def test_get_map_throttles_degenerate_leaders_only_build(tmp_path, monkeypatch):
    """A successful but leaders-only build must also be throttled - the
    map is stale (no followers) yet rebuilding on every poll would burst."""
    monkeypatch.setattr(mmmod, "MARKET_MAP_FILE", tmp_path / "m.json")
    mm = MarketMap()
    calls = {"n": 0}

    def degenerate_build():
        calls["n"] += 1
        return {"updated_at": datetime.now(timezone.utc).isoformat(),
                "universe_size": 2,
                "leaders": {"BTCUSDT": {"trend": "neutral"}},
                "symbols": {"BTCUSDT": {"leader": "BTCUSDT", "corr": 1.0,
                                         "is_leader": True}}}

    monkeypatch.setattr(mm, "_build", degenerate_build)
    mm.get_map()          # first call -> builds (degenerate)
    mm.get_map()          # stale (no followers) BUT throttled -> no rebuild
    assert calls["n"] == 1
    mm.get_map(force=True)  # force bypasses the throttle
    assert calls["n"] == 2


# ---------- v5.4: market cycle via leader coins ----------

def _stub_cycle_env(tmp_path, monkeypatch):
    monkeypatch.setattr(mmmod, "MARKET_MAP_FILE", tmp_path / "m.json")
    monkeypatch.setattr(mmmod, "MARKET_CYCLE_FILE", tmp_path / "c.json")
    monkeypatch.setattr(settings, "MARKET_GROUPS_FILE", str(tmp_path / "g.txt"))


def test_verdict_from_score_bands():
    assert MarketMap._verdict_from_score(70) == "strong_bullish"
    assert MarketMap._verdict_from_score(60) == "bullish"
    assert MarketMap._verdict_from_score(50) == "neutral"
    assert MarketMap._verdict_from_score(40) == "bearish"
    assert MarketMap._verdict_from_score(10) == "strong_bearish"


def test_market_cycle_verdicts_and_posture(tmp_path, monkeypatch):
    _stub_cycle_env(tmp_path, monkeypatch)
    mm = MarketMap()
    fake_map = _map_with_followers()
    fake_map["symbols"]["F4USDT"] = {
        "leader": INDEPENDENT, "corr": 0.2, "is_leader": False}
    monkeypatch.setattr(mm, "get_map", lambda force=False: fake_map)

    def fake_candles(sym, interval="1h", limit=200):
        if sym == "BTCUSDT":
            return _candles_df(drift=0.003, seed=3)    # strong uptrend
        return _candles_df(drift=-0.003, seed=9)       # strong downtrend

    monkeypatch.setattr(mmmod.data_fetcher, "get_candles", fake_candles)

    cycle = mm.run_market_cycle()
    btc = cycle["leaders"]["BTCUSDT"]
    eth = cycle["leaders"]["ETHUSDT"]
    assert btc["verdict"] in ("bullish", "strong_bullish")
    assert eth["verdict"] in ("bearish", "strong_bearish")
    assert btc["followers"] == 2 and eth["followers"] == 1
    for field in ("rsi14", "chg_24h_pct", "chg_48h_pct", "sma50_dist_pct",
                  "atr_pct", "score"):
        assert field in btc

    mk = cycle["market"]
    assert 0.0 <= mk["score"] <= 100.0
    assert mk["verdict"] == MarketMap._verdict_from_score(mk["score"])
    assert mk["posture_ar"]
    actions = {p["leader"]: p["action"] for p in mk["policy"]}
    assert actions["BTCUSDT"] == "prioritize_longs"
    assert actions["ETHUSDT"] == "avoid_longs"
    assert any(p["leader"] == INDEPENDENT for p in mk["policy"])

    # artifacts written
    assert (tmp_path / "c.json").exists()
    assert Path(settings.MARKET_GROUPS_FILE).exists()


def test_cycle_ttl_serves_cache_without_recompute(tmp_path, monkeypatch):
    _stub_cycle_env(tmp_path, monkeypatch)
    mm = MarketMap()
    fake_map = _map_with_followers()
    monkeypatch.setattr(mm, "get_map", lambda force=False: fake_map)
    calls = {"n": 0}

    def fake_candles(sym, interval="1h", limit=200):
        calls["n"] += 1
        return _candles_df(drift=0.003, seed=11)

    monkeypatch.setattr(mmmod.data_fetcher, "get_candles", fake_candles)

    first = mm.run_market_cycle()
    n_after_first = calls["n"]
    second = mm.run_market_cycle()  # within TTL -> served from cache
    assert calls["n"] == n_after_first
    assert second["updated_at"] == first["updated_at"]


def test_cycle_stale_when_old(tmp_path, monkeypatch):
    _stub_cycle_env(tmp_path, monkeypatch)
    stale = {"updated_at": (datetime.now(timezone.utc)
                            - timedelta(minutes=settings.MARKET_CYCLE_REFRESH_MIN + 5)
                            ).isoformat(),
             "leaders": {"BTCUSDT": {}},
             "market": {"score": 50}}
    assert MarketMap._cycle_is_stale(stale) is True
    fresh = dict(stale, updated_at=datetime.now(timezone.utc).isoformat())
    assert MarketMap._cycle_is_stale(fresh) is False
    assert MarketMap._cycle_is_stale(None) is True
    assert MarketMap._cycle_is_stale({}) is True


# ---------- v5.4: the Arabic decision file ----------

def test_groups_file_sorted_and_complete(tmp_path, monkeypatch):
    _stub_cycle_env(tmp_path, monkeypatch)
    mm = MarketMap()
    mm._map = _map_with_followers()  # F1 0.80 / F2 0.60 both follow BTC
    path = mm.save_groups_file(cycle=None)
    text = path.read_text(encoding="utf-8")
    assert "ملف تصنيف مجموعات السوق" in text
    assert "F1USDT" in text and "F2USDT" in text and "F3USDT" in text
    # followers sorted by corr (descending) inside their group block
    btc_block = text.split("مجموعة BTCUSDT")[1].split("مجموعة")[0]
    assert btc_block.index("F1USDT") < btc_block.index("F2USDT")
    assert "corr" in text and "beta" in text
    assert "قبل فتح صفقة شراء" in text  # usage guidance


def test_groups_file_includes_cycle_summary(tmp_path, monkeypatch):
    _stub_cycle_env(tmp_path, monkeypatch)
    mm = MarketMap()
    mm._map = _map_with_followers()
    posture = "بيئة صاعدة — يُفضل الشراء"
    cycle = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "leaders": {"BTCUSDT": {"verdict_ar": "صاعد", "score": 65, "rsi14": 61.2,
                                 "chg_24h_pct": 1.4, "chg_48h_pct": 2.1,
                                 "sma50_dist_pct": 1.1, "atr_pct": 0.9}},
        "market": {"score": 61, "verdict": "bullish", "verdict_ar": "صاعد",
                    "posture_ar": posture,
                    "policy": [{"leader": "BTCUSDT", "action": "prioritize_longs",
                                "followers": 2, "text_ar": "أولوية الشراء على تابعي BTCUSDT"}]},
    }
    path = mm.save_groups_file(cycle)
    text = path.read_text(encoding="utf-8")
    assert "حكم دورة السوق" in text
    assert posture in text
    assert "أولوية الشراء على تابعي BTCUSDT" in text
    assert "RSI14 61.2" in text


# ---------- API endpoints ----------

def test_api_market_map_includes_cycle(monkeypatch):
    from fastapi.testclient import TestClient
    from src.web.app import app
    from src.analysis.market_map import market_map

    ts = datetime.now(timezone.utc).isoformat()
    fake_view = {
        "updated_at": ts, "corr_threshold": 0.55, "lookback_hours": 168,
        "refresh_hours": 6,
        "leaders": {"BTCUSDT": {"trend": "bullish"}},
        "groups": {"BTCUSDT": [{"symbol": "F1USDT", "corr": 0.8, "beta": 1.1}]},
    }
    fake_cycle = {
        "updated_at": ts,
        "leaders": {"BTCUSDT": {"verdict": "bullish", "verdict_ar": "صاعد",
                                 "score": 65, "rsi14": 60, "chg_24h_pct": 1.2,
                                 "followers": 1}},
        "market": {"score": 65, "verdict": "bullish", "verdict_ar": "صاعد",
                    "posture_ar": "بيئة صاعدة", "policy": []},
    }
    monkeypatch.setattr(market_map, "get_map", lambda force=False: fake_view)
    monkeypatch.setattr(market_map, "grouped_view", lambda: fake_view)
    monkeypatch.setattr(market_map, "run_market_cycle", lambda force=False: fake_cycle)

    client = TestClient(app)
    r = client.get("/api/market-map")
    assert r.status_code == 200
    data = r.json()
    assert data["groups"]["BTCUSDT"][0]["symbol"] == "F1USDT"
    assert data["cycle"]["market"]["verdict"] == "bullish"
    assert data["cycle"]["leaders"]["BTCUSDT"]["followers"] == 1


def test_api_market_map_survives_cycle_failure(monkeypatch):
    from fastapi.testclient import TestClient
    from src.web.app import app
    from src.analysis.market_map import market_map

    fake_view = {"updated_at": None, "leaders": {}, "groups": {},
                  "refresh_hours": 6}
    monkeypatch.setattr(market_map, "get_map", lambda force=False: fake_view)
    monkeypatch.setattr(market_map, "grouped_view", lambda: fake_view)

    def broken_cycle(force=False):
        raise RuntimeError("no leader data")

    monkeypatch.setattr(market_map, "run_market_cycle", broken_cycle)
    client = TestClient(app)
    r = client.get("/api/market-map")
    assert r.status_code == 200
    assert r.json()["cycle"] is None


def test_api_groups_file_endpoint(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from src.web.app import app

    f = tmp_path / "g.txt"
    f.write_text("🧭 ملف تصنيف تجريبي\nBTCUSDT group\n", encoding="utf-8")
    monkeypatch.setattr(settings, "MARKET_GROUPS_FILE", str(f))
    client = TestClient(app)
    r = client.get("/api/market-groups-file")
    assert r.status_code == 200
    assert "تصنيف تجريبي" in r.text
