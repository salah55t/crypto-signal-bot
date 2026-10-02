"""
v5.29 ban-resilience tests.

Production 2026-10-02: a 1892s (escalated) Binance 418 IP ban on the shared
Render egress IP. The log showed the system behaving defensively (scalp
scanner skipping, cycle degrading) but exposed four workflow gaps:

  1. No POST-BAN PROBE: when a long ban expires, ALL callers (WS seeder +
     scalp tick + analysis cycle) fire into an IP that Binance may have
     silently EXTENDED the ban on - and every request during a ban extends
     it (the documented 598s -> 1892s escalation loop).
  2. Ban SOURCE invisible on /api/health (neighbor-driven vs our traffic).
  3. Scalp burns its REST weight (15 x weight-5 1s-klines per tick, the
     largest recurring REST cost) on a STATIC shortlist, most of it flat.
  4. exchangeInfo (weight 20, multi-MB) fetched per filter lookup, no cache.

Covered here:
  - rate_limiter: probe gate semantics (arm on real server bans only, fast
    bulk refusal after expiry, priority bypass, atomic probe claim),
    ban-source tracking, restored-ban arming.
  - binance_client: the /time probe flow (rejected -> re-armed cooldown +
    RateLimitError; accepted -> request proceeds) and the exchangeInfo
    6h cache (weight 20 paid once per TTL, force bypass).
  - scalp_scanner: the zero-REST WS momentum ring (sampling, move math,
    coverage rules) and mover-first shortlist ranking (long-only: dumpers
    rank below unknowns, positives jump the queue).

NOTE (mandatory project pattern): singletons are shared across the suite -
monkeypatch private state on them and let monkeypatch undo it.
"""
import importlib
import json
import os
import time
from collections import deque

import pytest


def _mod(name: str):
    return importlib.import_module(name)


# ----------------------------------------------------------------------
# Rate limiter: post-ban probe gate
# ----------------------------------------------------------------------
@pytest.fixture
def rl(tmp_path, monkeypatch):
    mod = _mod("src.core.rate_limiter")
    monkeypatch.setattr(mod, "STATE_FILE", tmp_path / "rate_state.json")
    limiter = mod.WeightedRateLimiter(budget_per_min=4500.0)
    return mod, limiter, tmp_path


def _expire(limiter):
    """Simulate the cooldown clock running out (tests never sleep)."""
    limiter._cooldown_until = time.monotonic() - 1.0


def test_probe_armed_by_418_and_blocks_bulk_after_expiry(rl):
    mod, limiter, _ = rl
    limiter.trigger_cooldown(900.0, source="418_ban")
    assert limiter.last_source() == "418_ban"
    assert limiter.armed_total() == 1
    # still inside the ban: needs_probe() is False (nothing to probe yet)
    assert limiter.needs_probe() is False

    _expire(limiter)
    # ban expired but unverified: bulk refused FAST, probe pending
    assert limiter.needs_probe() is True
    assert limiter.acquire(weight=2, timeout=0.5) is False
    # priority traffic (position watch) still passes - never blind
    assert limiter.acquire(weight=2, timeout=0.5, priority=True) is True

    # a verified-clean IP releases the herd
    limiter.finish_probe(True)
    assert limiter.needs_probe() is False
    assert limiter.acquire(weight=2, timeout=0.5) is True


def test_pressure_cooldowns_never_arm_the_probe(rl):
    mod, limiter, _ = rl
    limiter.trigger_cooldown(300.0, source="header_pressure")
    _expire(limiter)
    assert limiter.needs_probe() is False
    assert limiter.acquire(weight=2, timeout=0.5) is True


def test_legacy_unknown_source_keeps_old_behavior(rl):
    """Pre-v5.29 call sites (tests, tools) pass no source - no probe."""
    mod, limiter, _ = rl
    limiter.trigger_cooldown(598.0)  # default source="unknown"
    _expire(limiter)
    assert limiter.needs_probe() is False
    assert limiter.acquire(weight=2, timeout=0.5) is True


def test_begin_probe_is_atomic_and_recoverable(rl):
    mod, limiter, _ = rl
    limiter.trigger_cooldown(900.0, source="418_ban")
    _expire(limiter)
    assert limiter.begin_probe() is True
    assert limiter.begin_probe() is False  # exactly ONE claim
    # rejected probe frees the claim (client re-armed the cooldown itself)
    limiter.finish_probe(False)
    assert limiter._probe_required is True  # still unverified
    # simulate the re-armed ban the client just triggered, then its expiry
    limiter._cooldown_until = time.monotonic() + 300.0
    assert limiter.begin_probe() is False   # parked while the ban is live
    _expire(limiter)
    assert limiter.begin_probe() is True    # next tick may try again


def test_restored_ban_arms_the_probe(rl):
    mod, limiter, tmp_path = rl
    limiter.trigger_cooldown(1200.0, source="418_ban")
    f = tmp_path / "rate_state.json"
    payload = json.loads(f.read_text())
    payload["pid"] = os.getpid() + 777  # simulate a genuine restart
    f.write_text(json.dumps(payload))
    fresh = mod.WeightedRateLimiter(budget_per_min=4500.0)
    assert fresh.in_cooldown()
    assert fresh.last_source() == "restored"
    _expire(fresh)
    assert fresh.needs_probe() is True


def test_429_retry_source_arms_probe_only_for_long_bans(rl):
    mod, limiter, _ = rl
    limiter.trigger_cooldown(60.0, source="429_retry")  # short - no probe
    _expire(limiter)
    assert limiter.needs_probe() is False
    limiter.trigger_cooldown(600.0, source="429_retry")  # long - probe
    _expire(limiter)
    assert limiter.needs_probe() is True


# ----------------------------------------------------------------------
# Binance client: probe flow + exchangeInfo cache
# ----------------------------------------------------------------------
class _FakeResponse:
    def __init__(self, status=200, payload=None, headers=None):
        self.status_code = status
        self.text = "fake"
        self.headers = headers or {}
        self._payload = payload if payload is not None else {}

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


@pytest.fixture
def client(tmp_path, monkeypatch):
    import src.core.rate_limiter as rl_mod
    mod = _mod("src.core.binance_client")
    monkeypatch.setattr(
        rl_mod, "STATE_FILE", tmp_path / "rate_state.json")
    # snapshot & reset the shared singleton's ban state per test
    monkeypatch.setattr(rl_mod.rate_limiter, "_cooldown_until", 0.0)
    monkeypatch.setattr(rl_mod.rate_limiter, "_probe_required", False)
    monkeypatch.setattr(rl_mod.rate_limiter, "_probe_claimed", False)
    monkeypatch.setattr(rl_mod.rate_limiter, "_last_source", "none")
    monkeypatch.setattr(rl_mod.rate_limiter, "_armed_total", 0)
    cl = mod.binance_client
    cl._xinfo = None
    cl._xinfo_ts = 0.0
    return mod, cl, rl_mod


def test_client_probe_rejected_rearms_and_aborts(client, monkeypatch):
    mod, cl, rl_mod = client
    # arm a long ban as the 418 path would, then expire it (probe pending)
    rl_mod.rate_limiter.trigger_cooldown(900.0, source="418_ban")
    rl_mod.rate_limiter._cooldown_until = time.monotonic() - 1.0
    calls = {"n": 0}

    def _always_418(url, *a, **k):
        calls["n"] += 1
        return _FakeResponse(418, headers={"Retry-After": "1200"})

    monkeypatch.setattr(cl.session, "get", _always_418, raising=True)
    with pytest.raises(mod.RateLimitError):
        cl._get("/api/v3/ping")
    # the probe went out (ONE tiny request), was rejected, and the cooldown
    # was re-armed from the server value - the herd never fired
    assert calls["n"] == 1
    assert rl_mod.rate_limiter.in_cooldown()
    assert rl_mod.rate_limiter.last_source() == "probe_reject"
    assert rl_mod.rate_limiter.cooldown_remaining() >= 900.0
    assert rl_mod.rate_limiter.needs_probe() is False  # parked until expiry


def test_client_probe_ok_releases_the_request(client, monkeypatch):
    mod, cl, rl_mod = client
    rl_mod.rate_limiter.trigger_cooldown(900.0, source="418_ban")
    rl_mod.rate_limiter._cooldown_until = time.monotonic() - 1.0
    urls = []

    def _ok(url, *a, **k):
        urls.append(url)
        return _FakeResponse(200, payload={})

    monkeypatch.setattr(cl.session, "get", _ok, raising=True)
    out = cl._get("/api/v3/ping")
    assert out == {}
    assert rl_mod.rate_limiter.needs_probe() is False
    # one /time probe + the real request
    assert len([u for u in urls if u.endswith("/api/v3/time")]) == 1
    assert len(urls) == 2


def test_exchange_info_cached_weight20_paid_once(client, monkeypatch):
    mod, cl, rl_mod = client
    payload = {"symbols": [{
        "symbol": "BTCUSDT", "baseAsset": "BTC", "quoteAsset": "USDT",
        "filters": [
            {"filterType": "LOT_SIZE", "stepSize": "0.00001000",
             "minQty": "0.00001000", "maxQty": "1000"},
            {"filterType": "MIN_NOTIONAL", "minNotional": "10"},
            {"filterType": "PRICE_FILTER", "tickSize": "0.01000000"},
        ],
    }]}
    calls = {"n": 0}

    def _ok(url, *a, **k):
        calls["n"] += 1
        return _FakeResponse(200, payload=payload)

    monkeypatch.setattr(cl.session, "get", _ok, raising=True)
    info1 = cl.get_exchange_info()
    info2 = cl.get_exchange_info()          # cache hit
    assert info1 is info2
    assert calls["n"] == 1                  # weight 20 paid ONCE
    cl.get_exchange_info(force=True)        # explicit bypass
    assert calls["n"] == 2
    # filter lookups ride the same cache
    f = cl.get_symbol_filters("BTCUSDT")
    assert f["base_asset"] == "BTC"
    q = cl.round_quantity_to_lot("BTCUSDT", 0.1234567)
    assert calls["n"] == 2                  # still no extra HTTP
    assert q == pytest.approx(0.12345, abs=1e-9)


# ----------------------------------------------------------------------
# Scalp scanner: WS momentum ring + mover-first shortlist
# ----------------------------------------------------------------------
@pytest.fixture
def scalp(monkeypatch):
    mod = _mod("src.analysis.scalp_scanner")
    return mod.ScalpScanner()


def test_move_pct_requires_two_samples_and_span(scalp):
    now = time.time()
    scalp._ring_window_s = 300.0
    # old samples sit 10s inside the window (floating-point-safe margin)
    scalp._price_ring = {
        "ONEUSDT": deque([(now - 10, 100.0), (now, 101.0)]),      # too short
        "TWOUSDT": deque([(now - 100, 100.0), (now, 101.0)]),     # span < 150
        "OKUSDT": deque([(now - 290, 100.0), (now, 102.0)]),      # +2%
    }
    assert scalp._move_pct("ONEUSDT") is None
    assert scalp._move_pct("TWOUSDT") is None
    assert scalp._move_pct("OKUSDT") == pytest.approx(2.0)
    assert scalp._move_pct("GHOSTUSDT") is None


def test_rank_shortlist_movers_first_dumpers_last(scalp):
    now = time.time()
    scalp._ring_window_s = 300.0
    scalp._price_ring = {
        "BTCUSDT": deque([(now - 290, 100.0), (now, 102.5)]),   # +2.5%
        "SOLUSDT": deque([(now - 290, 100.0), (now, 97.0)]),    # dumper -3%
        "FLATUSDT": deque([(now - 290, 100.0), (now, 100.01)]), # ~flat
    }
    ranked = scalp._rank_shortlist(
        ["AAAUSDT", "SOLUSDT", "BTCUSDT", "FLATUSDT", "ETHUSDT"])
    # positive mover jumps the queue; unknowns + dumper + flat follow
    assert ranked[0] == "BTCUSDT"
    assert set(ranked[1:]) == {"SOLUSDT", "AAAUSDT", "FLATUSDT", "ETHUSDT"}
    assert ranked.index("BTCUSDT") < ranked.index("ETHUSDT")


def test_rank_shortlist_empty_ring_keeps_order(scalp):
    ranked = scalp._rank_shortlist(["CCCUSDT", "AAAUSDT", "BBBUSDT"])
    assert ranked == ["CCCUSDT", "AAAUSDT", "BBBUSDT"]


def test_sample_prices_populates_ring_zero_rest(scalp, monkeypatch):
    class _FakeWS:
        def get_live_prices(self, symbols, max_age_s=None):
            return {
                "BTCUSDT": "42000.5",
                "ETHUSDT": 2500.0,
                "BADUSDT": "not-a-number",
                "ZEROUSDT": 0.0,
            }

    import src.core.ws_feed as ws_mod
    monkeypatch.setattr(ws_mod, "ws_feed", _FakeWS())
    scalp._sample_prices(["BTCUSDT", "ETHUSDT", "BADUSDT", "ZEROUSDT"])
    assert "BTCUSDT" in scalp._price_ring
    assert "ETHUSDT" in scalp._price_ring
    assert "BADUSDT" not in scalp._price_ring
    assert "ZEROUSDT" not in scalp._price_ring
    ts, price = scalp._price_ring["BTCUSDT"][-1]
    assert price == pytest.approx(42000.5)
    assert abs(ts - time.time()) < 5.0
