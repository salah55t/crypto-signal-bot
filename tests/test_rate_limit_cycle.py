"""v5.2 rate-limit hardening: 418 ban backoff + cycle gate + health exposure.

Production incident (2026-09-24, Render shared egress IP): a 429/418 put the
limiter into cooldown, then run_analysis_cycle's ping() failed instantly
(acquire timeout 90s < remaining cooldown) and logged a MISLEADING
"Cannot reach Binance API - check network or VPN" error every tick.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from src.core.rate_limiter import WeightedRateLimiter, RateLimitError, rate_limiter


# ---------- limiter primitives ----------

def test_cooldown_remaining_zero_when_clear():
    lim = WeightedRateLimiter(budget_per_min=100)
    assert lim.cooldown_remaining() == 0.0
    assert lim.in_cooldown() is False


def test_long_cooldown_capped_at_24h():
    """v5.15: the sanity cap is now 24h (was 1h).

    Binance 418 Retry-After values for repeat offenders exceed 1h; the old
    1h cap made the bot re-poke the banned IP and EXTEND the ban. The 418
    call site pre-caps its own semantics (>= 900s, honouring the server);
    this is the last-resort bound only.
    """
    lim = WeightedRateLimiter(budget_per_min=100)
    lim.trigger_cooldown(200000)        # absurd value -> capped at 86400
    assert lim.in_cooldown() is True
    assert 86000 < lim.cooldown_remaining() <= 86400


def test_cooldown_only_extends_never_shortens():
    lim = WeightedRateLimiter(budget_per_min=100)
    lim.trigger_cooldown(600)
    first = lim.cooldown_remaining()
    lim.trigger_cooldown(5)             # shorter - must NOT shrink the window
    assert lim.cooldown_remaining() <= first
    assert lim.cooldown_remaining() > 500


# ---------- 418 IP-ban behavior ----------

class _FakeResponse:
    def __init__(self, status_code, headers=None, text=""):
        self.status_code = status_code
        self.headers = headers or {}
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.exceptions.HTTPError(f"{self.status_code}")

    def json(self):
        return {}


def _client():
    from src.core.binance_client import BinanceClient
    return BinanceClient()


def test_418_triggers_hard_backoff(monkeypatch):
    c = _client()
    monkeypatch.setattr(rate_limiter, "_cooldown_until", 0.0)  # clean state

    def fake_get(url, params=None, timeout=None):
        return _FakeResponse(418, headers={"Retry-After": "30"}, text="banned")

    monkeypatch.setattr(c.session, "get", fake_get)
    with pytest.raises(RateLimitError):
        c._get("/api/v3/ping")
    # IP ban -> at least 15 min, not the old 120s cap
    assert rate_limiter.cooldown_remaining() >= 890


def test_429_keeps_short_backoff(monkeypatch):
    c = _client()
    monkeypatch.setattr(rate_limiter, "_cooldown_until", 0.0)

    def fake_get(url, params=None, timeout=None):
        return _FakeResponse(429, headers={"Retry-After": "10"}, text="slow down")

    monkeypatch.setattr(c.session, "get", fake_get)
    with pytest.raises(RateLimitError):
        c._get("/api/v3/ping")
    remaining = rate_limiter.cooldown_remaining()
    assert 5 < remaining <= 125        # retry_after + 2, old behavior


# ---------- cycle gate ----------

def test_rate_limit_gate_skips_during_cooldown(monkeypatch):
    """v5.14: the gate is three-way ("run"/"degraded"/"skip").

    With the real singleton limiter and a DISABLED ws feed (or zero
    coverage), an active cooldown still skips the cycle exactly as before.
    """
    from src.core import cycle
    from config.settings import settings as _s
    monkeypatch.setattr(rate_limiter, "_cooldown_until", 0.0)  # reset
    monkeypatch.setattr(_s, "USE_WS_FEED", False)  # no WS fallback available
    assert cycle.rate_limit_gate() == "run"

    rate_limiter.trigger_cooldown(120)
    assert cycle.rate_limit_gate() == "skip"
    rate_limiter._cooldown_until = 0.0  # don't leak the cooldown into other test files


def test_health_exposes_rate_limit_state():
    from fastapi.testclient import TestClient
    from src.web.app import app
    client = TestClient(app)
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    rl = body.get("rate_limit") or {}
    assert "in_cooldown" in rl
    assert "cooldown_seconds_left" in rl
    assert "used_weight_1m" in rl


# ---------- v5.3: escalating shared-IP pressure ----------

def test_pressure_escalates_on_consecutive_hits(monkeypatch):
    lim = WeightedRateLimiter(budget_per_min=100)
    monkeypatch.setattr(lim, "_cooldown_until", 0.0)
    seen = []
    for _ in range(5):  # consecutive >=95% headers (neighbors hammering)
        lim.note_server_weight(5800)
        seen.append(lim.cooldown_remaining())
    # 30 -> 60 -> 120 -> 240 -> 300 (cap); never a fixed 30s poke-loop
    assert seen[0] <= 31
    assert seen[1] <= 61
    assert seen[2] <= 121
    assert seen[3] <= 241
    assert seen[4] <= 301
    assert seen[-1] >= 295                 # capped at PRESSURE_CAP_S
    assert len(set(seen)) >= 3             # genuinely escalating
    assert lim.pressure_streak() == 5


def test_pressure_85_band_escalates_from_10s(monkeypatch):
    lim = WeightedRateLimiter(budget_per_min=100)
    monkeypatch.setattr(lim, "_cooldown_until", 0.0)
    lim.note_server_weight(5300)           # 85-95% band -> base 10s
    assert 9 <= lim.cooldown_remaining() <= 11
    lim.note_server_weight(5300)
    assert 19 <= lim.cooldown_remaining() <= 21


def test_pressure_streak_resets_on_healthy_header(monkeypatch):
    lim = WeightedRateLimiter(budget_per_min=100)
    monkeypatch.setattr(lim, "_cooldown_until", 0.0)
    lim.note_server_weight(5800)
    lim.note_server_weight(5800)
    assert lim.pressure_streak() == 2
    lim.note_server_weight(1000)           # IP cooled down (< 80%)
    assert lim.pressure_streak() == 0
    # a NEW episode starts from the BASE again, not the escalated level
    # (cooldown extension semantics mean the old window may still run, but
    # the next REGISTERED cooldown is the base 30s)
    assert lim._register_pressure(30.0) <= 31.0


def test_pressure_cooldown_capped_below_418_ban(monkeypatch):
    lim = WeightedRateLimiter(budget_per_min=100)
    monkeypatch.setattr(lim, "_cooldown_until", 0.0)
    for _ in range(20):                    # extreme sustained pressure
        lim.note_server_weight(5999)
    assert lim.cooldown_remaining() <= 301  # header pressure != 418 hard ban


def test_health_exposes_ws_feed_state():
    from fastapi.testclient import TestClient
    from src.web.app import app
    client = TestClient(app)
    body = client.get("/api/health").json()
    ws = body.get("ws_feed") or {}
    assert "enabled" in ws
    assert "connected" in ws
    assert "cached_symbols" in ws


def test_klines_weight_scales_with_limit():
    from src.core.binance_client import _endpoint_weight
    assert _endpoint_weight("/api/v3/klines", {"limit": 100}) == 1
    assert _endpoint_weight("/api/v3/klines", {"limit": 200}) == 2
    assert _endpoint_weight("/api/v3/klines", {"limit": 1000}) == 5
    assert _endpoint_weight("/api/v3/klines", {"limit": 1500}) == 10
    assert _endpoint_weight("/api/v3/klines", {}) == 2  # default ~500
