"""v5.32 probe-backoff tests.

Production 2026-10-03 21:55 (post v5.31): the log showed the same 3-line
probe-reject burst with an ESCALATING Retry-After (1892s -> 2701s -> 3121s)
while /api/health simultaneously reported rest_spend_1h = 1 call / 80 weight
in the trailing hour and used_weight_1m = 0. Math: our traffic cannot earn a
418 on its own - the shared Render egress IP is banned by the herd. The only
poke WE still make while parked is the weight-1 /time verification probe at
each cooldown expiry, and poking at the exact Retry-After expiry again and
again reads as a repeat offense (Binance extends bans for violations DURING
a ban).

v5.32 change under test:
  - consecutive probe REJECTIONS add a growing margin ON TOP of the server
    Retry-After (base PROBE_BACKOFF_BASE_S, x2 per consecutive rejection,
    cap PROBE_BACKOFF_MAX_S) before the next probe goes out;
  - finish_probe(True) resets the streak (a verified-clean IP starts the
    next ban cycle from the base margin);
  - other cooldown sources (418_ban, 429_retry, header_pressure) are NOT
    touched - the server value governs there;
  - analyzer._ban_active / mtf._ban_active treat the PROBE WINDOW (cooldown
    expired, IP unverified) as ban-active so callers skip QUIETLY instead of
    raising through the client's probe gate (consistent with the v5.30
    zero-REST guard in get_batch_prices).

NOTE (mandatory project pattern): singletons are shared across the suite -
monkeypatch private state on them and let monkeypatch undo it; import via
importlib.import_module (src/core/__init__ rebinds singleton names, so
`import src.core.x as m` can return the INSTANCE, not the module).
"""
import importlib
import json
import os
import time

import pytest


def _mod(name: str):
    return importlib.import_module(name)


@pytest.fixture
def rl(tmp_path, monkeypatch):
    """Fresh limiter + deterministic backoff knobs (300s base, 3600s cap)."""
    mod = _mod("src.core.rate_limiter")
    from config.settings import settings as _settings
    monkeypatch.setattr(mod, "STATE_FILE", tmp_path / "rate_state.json")
    monkeypatch.setattr(_settings, "PROBE_BACKOFF_BASE_S", 300.0)
    monkeypatch.setattr(_settings, "PROBE_BACKOFF_MAX_S", 3600.0)
    limiter = mod.WeightedRateLimiter(budget_per_min=4500.0)
    return mod, limiter, tmp_path


def _expire(limiter):
    """Simulate the cooldown clock running out (tests never sleep)."""
    limiter._cooldown_until = time.monotonic() - 1.0


# ----------------------------------------------------------------------
# Margin escalation on consecutive probe rejections
# ----------------------------------------------------------------------
def test_first_rejection_adds_base_margin(rl):
    mod, limiter, _ = rl
    limiter.trigger_cooldown(2701.0, source="probe_reject")
    # server value + base margin (300s), not the server value alone
    assert 2995.0 <= limiter.cooldown_remaining() <= 2701.0 + 300.0 + 5.0
    assert limiter.probe_reject_streak() == 1
    assert limiter.last_source() == "probe_reject"


def test_consecutive_rejections_double_the_margin(rl):
    mod, limiter, _ = rl
    # rejection #1: server 2701 + 300
    limiter.trigger_cooldown(2701.0, source="probe_reject")
    first = limiter.cooldown_remaining()
    assert first == pytest.approx(3001.0, abs=5.0)
    # rejection #2 (a fresh probe after the previous cooldown expired):
    # server 3121 + 600 - the production escalation now costs us margin
    _expire(limiter)
    limiter.trigger_cooldown(3121.0, source="probe_reject")
    assert limiter.cooldown_remaining() == pytest.approx(3721.0, abs=5.0)
    assert limiter.probe_reject_streak() == 2
    # rejection #3: +1200
    _expire(limiter)
    limiter.trigger_cooldown(1200.0, source="probe_reject")
    assert limiter.cooldown_remaining() == pytest.approx(2400.0, abs=5.0)
    assert limiter.probe_reject_streak() == 3


def test_margin_capped_at_max(rl):
    mod, limiter, _ = rl
    for _ in range(10):  # base 300 * 2^9 = 153600 > cap 3600
        _expire(limiter)
        limiter.trigger_cooldown(100.0, source="probe_reject")
    assert limiter.probe_reject_streak() == 10
    # last arm: server 100 + capped margin 3600
    assert limiter.cooldown_remaining() == pytest.approx(3700.0, abs=5.0)


def test_probe_success_resets_the_streak(rl):
    mod, limiter, _ = rl
    limiter.trigger_cooldown(2701.0, source="probe_reject")
    _expire(limiter)
    limiter.trigger_cooldown(2701.0, source="probe_reject")
    assert limiter.probe_reject_streak() == 2
    # the IP finally answers - verified clean, fresh backoff next cycle
    limiter._probe_required = True
    limiter.finish_probe(True)
    assert limiter.probe_reject_streak() == 0
    _expire(limiter)
    limiter.trigger_cooldown(2701.0, source="probe_reject")
    assert limiter.cooldown_remaining() == pytest.approx(3001.0, abs=5.0)  # base again
    assert limiter.probe_reject_streak() == 1


def test_base_zero_disables_the_margin(rl, monkeypatch):
    mod, limiter, _ = rl
    from config.settings import settings as _settings
    monkeypatch.setattr(_settings, "PROBE_BACKOFF_BASE_S", 0.0)
    limiter.trigger_cooldown(2701.0, source="probe_reject")
    # streak still tracked (visibility), but the armed cooldown is exactly
    # the server value
    assert limiter.probe_reject_streak() == 1
    assert limiter.cooldown_remaining() == pytest.approx(2701.0, abs=5.0)


def test_other_sources_never_get_a_margin(rl):
    mod, limiter, _ = rl
    for source, seconds in (("418_ban", 900.0), ("429_retry", 1200.0),
                            ("header_pressure", 30.0), ("restored", 600.0)):
        _expire(limiter)
        limiter.trigger_cooldown(seconds, source=source)
        # the server value governs - no backoff margin outside probe_reject
        assert limiter.cooldown_remaining() == pytest.approx(seconds, abs=2.0)
    assert limiter.probe_reject_streak() == 0


def test_margin_survives_a_restart_via_state_file(rl):
    mod, limiter, tmp_path = rl
    limiter.trigger_cooldown(2701.0, source="probe_reject")
    f = tmp_path / "rate_state.json"
    payload = json.loads(f.read_text())
    armed_s = payload["last_seconds"]
    # the persisted cooldown is server + margin, not the bare server value
    assert armed_s == pytest.approx(3001.0, abs=5.0)
    payload["pid"] = os.getpid() + 777  # simulate a genuine restart
    f.write_text(json.dumps(payload))
    fresh = mod.WeightedRateLimiter(budget_per_min=4500.0)
    # the restarted process inherits the FULL armed wait (server + margin)
    assert fresh.cooldown_remaining() == pytest.approx(3001.0, abs=10.0)


# ----------------------------------------------------------------------
# The probe window is ban-active for the analyzer/mtf guards
# ----------------------------------------------------------------------
@pytest.fixture
def parked_shared_limiter(monkeypatch):
    """Force the SHARED limiter into the probe window (expired, unverified)."""
    mod = _mod("src.core.rate_limiter")
    rl_singleton = mod.rate_limiter
    monkeypatch.setattr(rl_singleton, "_cooldown_until",
                        time.monotonic() - 1.0)
    monkeypatch.setattr(rl_singleton, "_probe_required", True)
    monkeypatch.setattr(rl_singleton, "_probe_claimed", False)
    return mod, rl_singleton


def test_analyzer_ban_active_includes_probe_window(parked_shared_limiter):
    _mod_analyzer = _mod("src.analysis.analyzer")
    assert _mod_analyzer._ban_active() is True
    # and once the probe verifies the IP, the window opens again
    _, rl_singleton = parked_shared_limiter
    rl_singleton._probe_required = False
    assert _mod_analyzer._ban_active() is False


def test_mtf_ban_active_includes_probe_window(parked_shared_limiter):
    _mod_mtf = _mod("src.analysis.mtf")
    assert _mod_mtf._ban_active() is True
    _, rl_singleton = parked_shared_limiter
    rl_singleton._probe_required = False
    assert _mod_mtf._ban_active() is False


def test_analyzer_ban_active_still_uses_cooldown_threshold(monkeypatch):
    """No regression: below the threshold (and no probe) it stays False."""
    mod = _mod("src.core.rate_limiter")
    rl_singleton = mod.rate_limiter
    monkeypatch.setattr(rl_singleton, "_cooldown_until",
                        time.monotonic() + 10.0)  # 10s left < both thresholds
    monkeypatch.setattr(rl_singleton, "_probe_required", False)
    analyzer = _mod("src.analysis.analyzer")
    mtf = _mod("src.analysis.mtf")
    assert analyzer._ban_active() is False
    assert mtf._ban_active() is False
    # a long cooldown still trips both guards
    monkeypatch.setattr(rl_singleton, "_cooldown_until",
                        time.monotonic() + 600.0)
    assert analyzer._ban_active() is True
    assert mtf._ban_active() is True


# ----------------------------------------------------------------------
# /api/health visibility
# ----------------------------------------------------------------------
def test_probe_reject_streak_accessor_reads_shared_state():
    mod = _mod("src.core.rate_limiter")
    assert isinstance(mod.rate_limiter.probe_reject_streak(), int)
