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


@pytest.fixture(autouse=True)
def _neutral_regime_policy(monkeypatch):
    """v5.18: isolate tests from a locally-written data/regime_state.json.

    A diagnostic cycle (or production checkout) can persist a trending_bear
    policy whose min_confidence_adjust=+3 raises the regime-adjusted
    confidence gate above static MIN_CONFIDENCE - ledger/veteran tests then
    see recs silently rejected with 'Confidence too low (70% < 71%)'.
    Tests that need a specific policy inject their own router (see
    test_v513 _FixedRouter), so defaulting the real singleton to a neutral
    policy is safe everywhere.

    NOTE: src.analysis re-exports singletons under the module name, so
    `import src.analysis.regime_router as X` yields the INSTANCE, not the
    module - use importlib.import_module (project test convention).
    """
    try:
        import importlib
        _rrmod = importlib.import_module("src.analysis.regime_router")
        monkeypatch.setattr(_rrmod.regime_router, "active_policy",
                            lambda: {}, raising=False)
    except Exception:
        pass
    yield


@pytest.fixture(autouse=True)
def _strip_singleton_static_shadows():
    """v5.31: undo pytest's staticmethod instance-shadow leak on singletons.

    `monkeypatch.setattr(single_instance, "some_staticmethod", fake)` records
    the old value via getattr() (the plain function) and restores it with
    setattr() ON THE INSTANCE - permanently SHADOWING the class-level
    staticmethod with an instance attribute. The shadow holds the original
    function so the leak was behavior-invisible for months, but it silently
    takes precedence over any later class-level patch (v5.31 scalp tests
    patched DataFetcher.get_candles on the class and never saw it fire).
    After every test, drop instance attrs that merely shadow the class -
    legit instance state (_ob_cache etc.) is not defined on the class and
    survives untouched.
    """
    yield
    import importlib
    try:
        dfm = importlib.import_module("src.core.data_fetcher")
        inst = dfm.data_fetcher
        cls = type(inst)
        for name in list(vars(inst)):
            if name in vars(cls) or any(
                    name in vars(k) for k in cls.__mro__[1:]):
                delattr(inst, name)
    except Exception:
        pass
