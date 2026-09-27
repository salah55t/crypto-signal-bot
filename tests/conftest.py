"""
Shared pytest fixtures.

v5.15+: the rate limiter persists REST-ban cooldowns to data/rate_state.json
and restores them in genuinely NEW processes (Render deploys). A pytest run
IS a new process, so without isolation a stale file from a previous run
(including files written by tests that trigger 429/418 paths) would be
restored into the singleton and poison cooldown-sensitive tests. Every test
therefore sees its own throwaway state file.
"""
import pytest


@pytest.fixture(autouse=True)
def _isolated_rate_state(tmp_path, monkeypatch):
    import src.core.rate_limiter as rl
    monkeypatch.setattr(rl, "STATE_FILE", tmp_path / "rate_state.json")
    yield
