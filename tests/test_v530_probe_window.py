"""
v5.30 probe-window discipline tests.

Production 2026-10-03 (the log that triggered this revision): a probe
REJECT re-armed a 2701s cooldown AND, one second earlier, a header-pressure
hit showed the shared IP was still hot. The system parked correctly, but
one poke vector remained: the PROBE WINDOW (cooldown expired, IP not yet
verified). Priority callers (dashboard P&L poll, position-watch gap-fill)
bypass the probe gate inside the client, so a single dashboard poll with a
WS gap would send real requests into a still-418 IP and EXTEND the ban -
the exact repeat-offense loop the log warned about ("ban likely EXTENDED -
repeat offense").

Covered here:
  - data_fetcher.get_batch_prices: REST gap-fill is SKIPPED during the
    probe window (WS + last-known serve instead); REST returns once the
    probe clears the IP.
  - binance_client: the BINANCE_PROXY_URL escape hatch wires session
    proxies (dedicated-IP egress ends herd-driven ban cycles); default
    stays direct.
"""
import importlib
import time

import pytest


def _mod(name: str):
    # NOTE: src/core/__init__ rebinds the singleton NAME (e.g.
    # `binance_client` the instance) in the package namespace, so plain
    # `import src.core.binance_client as m` can return the INSTANCE -
    # importlib.import_module always returns the real module.
    return importlib.import_module(name)


@pytest.fixture
def rl(tmp_path, monkeypatch):
    import src.core.rate_limiter as rl_mod
    monkeypatch.setattr(rl_mod, "STATE_FILE", tmp_path / "rate_state.json")
    # snapshot & reset the shared singleton's ban state per test
    monkeypatch.setattr(rl_mod.rate_limiter, "_cooldown_until", 0.0)
    monkeypatch.setattr(rl_mod.rate_limiter, "_probe_required", False)
    monkeypatch.setattr(rl_mod.rate_limiter, "_probe_claimed", False)
    monkeypatch.setattr(rl_mod.rate_limiter, "_last_source", "none")
    monkeypatch.setattr(rl_mod.rate_limiter, "_armed_total", 0)
    return rl_mod


@pytest.fixture
def no_ws(monkeypatch):
    """WS feed serves NOTHING (worst case: full gap, REST would be asked)."""
    import src.core.ws_feed as ws_mod

    class _EmptyWS:
        def get_live_prices(self, symbols, max_age_s=None):
            return {}

    monkeypatch.setattr(ws_mod, "ws_feed", _EmptyWS())


def _arm_expired_ban(rl_mod):
    """418 ban armed then expired -> exactly the probe window."""
    rl_mod.rate_limiter.trigger_cooldown(900.0, source="418_ban")
    rl_mod.rate_limiter._cooldown_until = time.monotonic() - 1.0
    assert rl_mod.rate_limiter.needs_probe() is True


def test_gap_fill_zero_rest_during_probe_window(rl, no_ws, monkeypatch):
    from src.core.data_fetcher import DataFetcher
    bc_mod = _mod("src.core.binance_client")

    _arm_expired_ban(rl)
    calls = {"n": 0}

    def _must_not_call(url, *a, **k):
        calls["n"] += 1
        raise AssertionError(
            f"REST poked during probe window: {url}")

    monkeypatch.setattr(bc_mod.binance_client.session, "get",
                        _must_not_call, raising=True)
    out = DataFetcher.get_batch_prices(["BTCUSDT"], priority=True)
    assert out == {}
    assert calls["n"] == 0  # zero HTTP - the ban can no longer be extended
    # the limiter state is untouched: still unverified, still parked
    assert rl.rate_limiter.needs_probe() is True


def test_gap_fill_rest_returns_after_probe_clears(rl, no_ws, monkeypatch):
    from src.core.data_fetcher import DataFetcher
    bc_mod = _mod("src.core.binance_client")

    _arm_expired_ban(rl)
    rl.rate_limiter.finish_probe(True)  # the /time probe verified the IP
    assert rl.rate_limiter.needs_probe() is False

    calls = {"n": 0}

    def _ok(url, *a, **k):
        calls["n"] += 1
        return type("R", (), {
            "status_code": 200, "text": "ok",
            "headers": {"X-MBX-USED-WEIGHT-1M": "10"},
            "raise_for_status": lambda self: None,
            "json": lambda self: [{"symbol": "BTCUSDT",
                                   "lastPrice": "42000.5"}],
        })()

    monkeypatch.setattr(bc_mod.binance_client.session, "get",
                        _ok, raising=True)
    out = DataFetcher.get_batch_prices(["BTCUSDT"], priority=True)
    assert calls["n"] >= 1
    assert out.get("BTCUSDT") == pytest.approx(42000.5)


def test_proxy_escape_hatch_wires_dedicated_egress(rl, monkeypatch):
    from config.settings import settings
    bc_mod = _mod("src.core.binance_client")

    monkeypatch.setattr(settings, "BINANCE_PROXY_URL",
                        "http://user:pass@203.0.113.7:3128")
    cl = bc_mod.BinanceClient()
    assert cl.session.proxies == {
        "http": "http://user:pass@203.0.113.7:3128",
        "https": "http://user:pass@203.0.113.7:3128",
    }

    monkeypatch.setattr(settings, "BINANCE_PROXY_URL", "")
    cl_direct = bc_mod.BinanceClient()
    assert not cl_direct.session.proxies  # default: direct connection
