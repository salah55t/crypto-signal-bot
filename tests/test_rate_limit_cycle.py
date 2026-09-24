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


def test_long_cooldown_capped_at_1h():
    lim = WeightedRateLimiter(budget_per_min=100)
    lim.trigger_cooldown(7200)          # absurd value -> capped
    assert lim.in_cooldown() is True
    assert 3500 < lim.cooldown_remaining() <= 3600


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
    from src.core import cycle
    monkeypatch.setattr(rate_limiter, "_cooldown_until", 0.0)  # reset
    assert cycle.rate_limit_gate() is False

    rate_limiter.trigger_cooldown(120)
    assert cycle.rate_limit_gate() is True


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
