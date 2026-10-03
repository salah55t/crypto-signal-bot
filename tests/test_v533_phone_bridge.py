"""
v5.33 phone-bridge route tests.

The user's structural fix for herd-driven 418 bans: route the bot's
Binance-bound REST egress through THEIR OWN phone (personal mobile/WiFi
IP) instead of Render's shared pool - the phone acts as a Tailscale
exit-node, and BINANCE_PROXY_URL points at the userspace SOCKS hop
(socks5h://127.0.0.1:1055) that tailscaled exposes on Render.

A phone, however, sleeps. The route therefore must be MANAGED:

  - transport-level failures (ProxyError/ConnectionError) flip REST to
    DIRECT egress for BINANCE_PROXY_RETRY_S, then ONE in-band re-probe
    (weight-0 /ping through the proxy) flips it back when the phone
    returns - the bot never dies because the phone did;
  - 418/429 ban responses are NOT transport errors and NEVER flip the
    route - a banned phone IP must not poison the shared one (falling
    back "to escape a ban" would just earn the Render IP its own ban);
  - the probe itself (_send_probe) verifies whichever route is active;
  - /api/health exposes the live route via proxy_state().
"""
import importlib
import time

import pytest
import requests

from src.core.rate_limiter import RateLimitError


def _mod(name: str):
    # src/core/__init__ rebinds singleton NAMES - importlib returns the
    # real module (see test_v530_probe_window.py for the full story).
    return importlib.import_module(name)


PROXY = "socks5h://127.0.0.1:1055"
ROUTE = {"http": PROXY, "https": PROXY}


@pytest.fixture
def rl(tmp_path, monkeypatch):
    import src.core.rate_limiter as rl_mod
    monkeypatch.setattr(rl_mod, "STATE_FILE", tmp_path / "rate_state.json")
    monkeypatch.setattr(rl_mod.rate_limiter, "_cooldown_until", 0.0)
    monkeypatch.setattr(rl_mod.rate_limiter, "_probe_required", False)
    monkeypatch.setattr(rl_mod.rate_limiter, "_probe_claimed", False)
    monkeypatch.setattr(rl_mod.rate_limiter, "_last_source", "none")
    monkeypatch.setattr(rl_mod.rate_limiter, "_armed_total", 0)
    return rl_mod


@pytest.fixture
def client(rl, monkeypatch):
    """Fresh BinanceClient wired to the phone-bridge route."""
    from config.settings import settings
    bc_mod = _mod("src.core.binance_client")
    monkeypatch.setattr(settings, "BINANCE_PROXY_URL", PROXY)
    monkeypatch.setattr(settings, "BINANCE_PROXY_RETRY_S", 300.0)
    return bc_mod, bc_mod.BinanceClient()


class FakeSession:
    """Scripted session - each call pops one step(url, proxies)."""

    def __init__(self, *steps):
        self.steps = list(steps)
        self.calls = []  # (method, url, proxies)
        self.headers = {}

    def _next(self, method, url, kw):
        proxies = kw.get("proxies")
        self.calls.append((method, url, proxies))
        step = self.steps.pop(0) if self.steps else self.steps[-1]
        # steps may be pre-built responses OR callables(url, proxies)
        return step(url, proxies) if callable(step) else step

    def get(self, url, **kw):
        return self._next("get", url, kw)

    def post(self, url, **kw):
        return self._next("post", url, kw)


def _resp(status=200, retry_after=None, payload=None):
    headers = {"X-MBX-USED-WEIGHT-1M": "5"}
    if retry_after is not None:
        headers["Retry-After"] = str(retry_after)

    def _raise_for_status(self):
        if status >= 400:
            raise requests.exceptions.HTTPError(f"HTTP {status}",
                                                response=self)

    return type("R", (), {
        "status_code": status,
        "headers": headers,
        "text": "ok" if status == 200 else "err",
        "json": lambda self: payload if payload is not None else {},
        "raise_for_status": _raise_for_status,
    })()


def _conn_err(url, proxies):
    raise requests.exceptions.ConnectionError("phone asleep")


def _mark_down(cl, window_elapsed=True):
    cl._proxy_healthy = False
    cl._proxy_retry_at = (time.monotonic() - 1.0 if window_elapsed
                          else time.monotonic() + 300.0)


# ---------------------------------------------------------------------------
# routing basics
# ---------------------------------------------------------------------------

def test_no_proxy_configured_stays_direct(rl, monkeypatch):
    from config.settings import settings
    bc_mod = _mod("src.core.binance_client")
    monkeypatch.setattr(settings, "BINANCE_PROXY_URL", "")
    cl = bc_mod.BinanceClient()
    assert cl._effective_proxies() is None

    fake = FakeSession(_resp())
    cl.session = fake
    cl._route("get", "https://x/api/v3/time")
    assert fake.calls[0][2] is None  # proxies=None -> direct egress


def test_healthy_route_sends_proxy_per_request(client):
    bc_mod, cl = client
    fake = FakeSession(_resp())
    cl.session = fake
    cl._route("get", "https://x/api/v3/time")
    assert fake.calls[0][2] == ROUTE


def test_phone_asleep_one_direct_retry_then_park(client):
    bc_mod, cl = client
    fake = FakeSession(_conn_err, _resp())
    cl.session = fake

    out = cl._route("get", "https://x/api/v3/klines")
    assert out.status_code == 200  # the direct retry saved the tick
    # call 0 rode the proxy and died; call 1 rode direct
    assert fake.calls[0][2] == ROUTE
    assert fake.calls[1][2] is None
    # state flipped, with the reason recorded for /api/health
    st = cl.proxy_state()
    assert st["using"] == "direct-fallback"
    assert st["healthy"] is False
    assert "phone asleep" in st["last_fail_reason"]

    # inside the retry window: zero proxy pings, straight direct
    fake = FakeSession(_resp())
    cl.session = fake
    cl._route("get", "https://x/api/v3/klines")
    assert len(fake.calls) == 1
    assert fake.calls[0][2] is None
    assert all("ping" not in c[1] for c in fake.calls)


def test_direct_retry_also_failing_propagates(client):
    bc_mod, cl = client
    cl.session = FakeSession(_conn_err, _conn_err)
    with pytest.raises(requests.exceptions.ConnectionError):
        cl._route("get", "https://x/api/v3/klines")
    assert cl.proxy_state()["using"] == "direct-fallback"


# ---------------------------------------------------------------------------
# the golden rule: ban errors NEVER flip the route
# ---------------------------------------------------------------------------

def test_418_ban_via_phone_does_not_fall_back_to_shared_ip(rl, client):
    bc_mod, cl = client
    cl.session = FakeSession(_resp(418, retry_after=900))
    with pytest.raises(RateLimitError):
        cl._get("/api/v3/time")
    st = cl.proxy_state()
    assert st["healthy"] is True
    assert st["using"] == "proxy"  # route untouched - no shared-IP poisoning
    assert st["last_fail_reason"] is None


def test_429_via_phone_does_not_fall_back_either(rl, client):
    bc_mod, cl = client
    cl.session = FakeSession(_resp(429, retry_after=30))
    with pytest.raises(RateLimitError):
        cl._get("/api/v3/time")
    assert cl.proxy_state()["using"] == "proxy"


# ---------------------------------------------------------------------------
# the phone came back: one in-band re-probe restores the route
# ---------------------------------------------------------------------------

def test_reprobe_success_restores_phone_route(client):
    bc_mod, cl = client
    _mark_down(cl, window_elapsed=True)
    # step 1 = the weight-0 /ping THROUGH the proxy, step 2 = real traffic
    fake = FakeSession(lambda u, p: _resp(200), _resp())
    cl.session = fake

    out = cl._route("get", "https://x/api/v3/klines")
    assert out.status_code == 200
    assert fake.calls[0][1].endswith("/api/v3/ping")
    assert fake.calls[0][2] == ROUTE
    assert fake.calls[1][2] == ROUTE  # real traffic back on the phone
    st = cl.proxy_state()
    assert st["using"] == "proxy" and st["healthy"] is True
    assert st["last_fail_reason"] is None and st["retry_in_s"] == 0


def test_reprobe_failure_extends_direct_window(client):
    bc_mod, cl = client
    _mark_down(cl, window_elapsed=True)
    fake = FakeSession(_conn_err, _resp())
    cl.session = fake

    out = cl._route("get", "https://x/api/v3/klines")
    assert out.status_code == 200  # direct served the tick
    assert fake.calls[0][1].endswith("/api/v3/ping")  # probe tried first
    assert fake.calls[1][2] is None
    st = cl.proxy_state()
    assert st["using"] == "direct-fallback"
    assert 0 <= st["retry_in_s"] <= 300  # window re-armed


# ---------------------------------------------------------------------------
# the post-ban probe verifies the ACTIVE route (not a hardcoded one)
# ---------------------------------------------------------------------------

def test_send_probe_rides_the_active_route(rl, client):
    bc_mod, cl = client
    cl.session = FakeSession(_resp())
    assert cl._send_probe() is True
    assert cl.session.calls[0][1].endswith("/api/v3/time")
    assert cl.session.calls[0][2] == ROUTE

    # phone asleep with the window still open -> probe must ride DIRECT
    # (verifying the route that will actually receive live traffic)
    _mark_down(cl, window_elapsed=False)
    cl.session = FakeSession(_resp())
    assert cl._send_probe() is True
    assert cl.session.calls[0][2] is None


def test_health_exposes_phone_bridge_state(rl, monkeypatch):
    from config.settings import settings
    bc_mod = _mod("src.core.binance_client")
    monkeypatch.setattr(settings, "BINANCE_PROXY_URL", PROXY)
    monkeypatch.setattr(settings, "BINANCE_PROXY_RETRY_S", 300.0)
    cl = bc_mod.BinanceClient()
    st = cl.proxy_state()
    assert st == {
        "configured": True, "scheme": "socks5h", "healthy": True,
        "using": "proxy", "retry_in_s": 0, "last_fail_reason": None,
    }
