"""v5.27 - Strategy tuning from LIVE results (first paper day on Render):
10 closed trades, 2W/8L, driven by three structural fee defects:

  * micro_scalp 6/6 losses (-1.67%): fees were 72% of the loss - the old
    0.02% ATR floor admitted markets that cannot pay the 0.2% round-trip
    fee inside a 60s hold (MFE never exceeded 0.077%).
    -> fee-linked vol floors + fee-linked TP1 floor (tested in
       test_v525_scalp.py, fixture-level).
  * double_indicator 1/1 loss (ZEC -0.75%): TP1 0.50% vs SL 0.55% -
    inverted geometry, breakeven win rate 71% after fees.
    -> TP1 floor 3.5x fees + TP1 ATR multiple 1.8x > SL 1.5x.
  * bottom_scanner 2W/1L but the loser gave back the whole bounce (DOGE
    peaked +0.46%, exited -0.295% net): nothing locks profit below the
    ladder's +1% first rung.
    -> global fee-survival rung in apply_trailing_logic (this file).

Covers:
  * the fee-survival rung: trigger math (MFE peak + give-back + sub-ladder
    profit), SL lock at entry + round-trip fees, hard-cap interplay,
  * no-trigger cases (MFE too small, no give-back, ladder zone, disabled),
  * ladder precedence (the rung can never loosen a tighter ladder lock),
  * bearish mirror,
  * scalp exemption (fixed-hold positions are never trailed),
  * double_indicator TP1 fee geometry (floor + RR-on-TP1 >= 1.2).
"""
import importlib

import pytest

cfgmod = importlib.import_module("config.settings")
settings = cfgmod.settings
manager_module = importlib.import_module("src.risk.manager")

from tests.test_v5_veteran import make_manager, make_position


FEE_RT = 2.0 * settings.TRADING_FEE_PCT  # 0.2% at the default schedule
LOCK_PCT = FEE_RT + settings.FEE_SURVIVAL_NET_PCT  # 0.22%


# ------------------------------------------------------------------
# Fee-survival rung (bullish)
# ------------------------------------------------------------------
def test_fee_survival_locks_breakeven_after_giveback(tmp_path, monkeypatch):
    """DOGE replay: peak +0.75%, price falls back to +0.5% (gave back
    0.25% >= GIVEBACK_PCT) while still below the +1% ladder rung ->
    SL jumps to entry + round-trip fees (+ NET_PCT)."""
    monkeypatch.setattr(settings, "STRUCTURAL_EXITS_ENABLED", False)
    rm = make_manager(tmp_path, monkeypatch)
    pos = make_position(entry=100.0, sl=98.0, tp=104.0)
    pos["mfe_pct"] = 0.75
    pos["peak_price"] = 100.75
    rm.open_positions = [pos]
    updates = rm.apply_trailing_logic({"TESTUSDT": 100.5})
    assert updates, "fee-survival rung must produce an update"
    u = updates[-1]
    assert u["new_sl"] == pytest.approx(100.0 * (1 + LOCK_PCT / 100.0))
    assert "Fee-survival" in u["reason"]
    assert rm.open_positions[0]["stop_loss"] == pytest.approx(
        100.0 * (1 + LOCK_PCT / 100.0))


def test_fee_survival_not_armed_below_mfe_trigger(tmp_path, monkeypatch):
    rm = make_manager(tmp_path, monkeypatch)
    pos = make_position(entry=100.0, sl=98.0, tp=104.0)
    pos["mfe_pct"] = settings.FEE_SURVIVAL_MFE_PCT - 0.05  # just under
    pos["peak_price"] = 100.40
    rm.open_positions = [pos]
    updates = rm.apply_trailing_logic({"TESTUSDT": 100.2})  # gave back 0.2%
    assert updates == []
    assert rm.open_positions[0]["stop_loss"] == 98.0


def test_fee_survival_not_armed_without_giveback(tmp_path, monkeypatch):
    """Still making new highs - do not strangle a running trade."""
    rm = make_manager(tmp_path, monkeypatch)
    pos = make_position(entry=100.0, sl=98.0, tp=104.0)
    pos["mfe_pct"] = 0.60
    pos["peak_price"] = 100.60
    rm.open_positions = [pos]
    updates = rm.apply_trailing_logic({"TESTUSDT": 100.55})  # gave back 0.05
    assert updates == []


def test_fee_survival_silent_inside_ladder_zone(tmp_path, monkeypatch):
    """At/above +1% the ladder owns the trade - the rung must not fire."""
    rm = make_manager(tmp_path, monkeypatch)
    pos = make_position(entry=100.0, sl=98.0, tp=104.0)
    pos["mfe_pct"] = 1.40
    pos["peak_price"] = 101.40
    rm.open_positions = [pos]
    updates = rm.apply_trailing_logic({"TESTUSDT": 101.15})  # +1.15%, gave 0.25
    assert updates, "ladder lock1 must fire here"
    assert all("Fee-survival" not in u["reason"] for u in updates)
    # ladder locked +0.30%, not the fee lock +0.22%
    assert rm.open_positions[0]["stop_loss"] == pytest.approx(100.30)


def test_fee_survival_never_loosens_tighter_sl(tmp_path, monkeypatch):
    rm = make_manager(tmp_path, monkeypatch)
    pos = make_position(entry=100.0, sl=100.40, tp=104.0)  # already tight
    pos["mfe_pct"] = 0.75
    pos["peak_price"] = 100.75
    rm.open_positions = [pos]
    updates = rm.apply_trailing_logic({"TESTUSDT": 100.5})
    # lock (100.22) < current SL (100.40) -> no update may lower the SL
    assert all(u["new_sl"] is None or u["new_sl"] >= 100.40 for u in updates)
    assert rm.open_positions[0]["stop_loss"] == 100.40


def test_fee_survival_capped_below_current_price(tmp_path, monkeypatch):
    """A deep give-back can put the lock above the market - the hard cap
    must keep it at (current - 0.1%) instead of an instant stop-out."""
    rm = make_manager(tmp_path, monkeypatch)
    pos = make_position(entry=100.0, sl=98.0, tp=104.0)
    pos["mfe_pct"] = 0.75
    pos["peak_price"] = 100.75
    rm.open_positions = [pos]
    updates = rm.apply_trailing_logic({"TESTUSDT": 100.21})  # gave back 0.54
    assert updates
    u = updates[-1]
    assert u["new_sl"] == pytest.approx(100.21 * 0.999)
    assert u["new_sl"] < 100.0 * (1 + LOCK_PCT / 100.0)


def test_fee_survival_disabled(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "FEE_SURVIVAL_ENABLED", False)
    rm = make_manager(tmp_path, monkeypatch)
    pos = make_position(entry=100.0, sl=98.0, tp=104.0)
    pos["mfe_pct"] = 0.75
    pos["peak_price"] = 100.75
    rm.open_positions = [pos]
    updates = rm.apply_trailing_logic({"TESTUSDT": 100.5})
    assert updates == []


def test_fee_survival_bearish_mirror(tmp_path, monkeypatch):
    rm = make_manager(tmp_path, monkeypatch)
    pos = make_position(direction="bearish", entry=100.0, sl=102.0, tp=96.0)
    pos["mfe_pct"] = 0.75
    pos["trough_price"] = 99.25
    rm.open_positions = [pos]
    updates = rm.apply_trailing_logic({"TESTUSDT": 99.5})  # +0.5% for a short
    assert updates, "bearish fee-survival rung must produce an update"
    u = updates[-1]
    assert u["new_sl"] == pytest.approx(100.0 * (1 - LOCK_PCT / 100.0))
    assert "Fee-survival" in u["reason"]


def test_fee_survival_skips_scalp_positions(tmp_path, monkeypatch):
    """Fixed-hold scalps die at the time exit by design - the watcher has
    no mid-hold tick, so the rung must stay out of their way."""
    rm = make_manager(tmp_path, monkeypatch)
    pos = make_position(entry=100.0, sl=99.2, tp=100.35)
    pos["boosted_from_scalp"] = True
    pos["mfe_pct"] = 0.75
    pos["peak_price"] = 100.75
    rm.open_positions = [pos]
    updates = rm.apply_trailing_logic({"TESTUSDT": 100.5})
    assert updates == []
    assert rm.open_positions[0]["stop_loss"] == 99.2


# ------------------------------------------------------------------
# double_indicator TP1 fee geometry (the ZEC fix)
# ------------------------------------------------------------------
def test_double_ind_tp1_clears_fees_and_dominates_sl():
    from tests.test_v522_double_indicator import _burst_df
    from src.strategies.double_indicator_strategy import DoubleIndicatorStrategy
    sig = DoubleIndicatorStrategy().analyze(_burst_df(), "MOMUSDT")
    assert sig.direction == "bullish"
    d = sig.details
    tp1_floor = max(settings.DOUBLE_IND_TP1_MIN_PCT,
                    2.0 * settings.TRADING_FEE_PCT
                    * settings.DOUBLE_IND_FEE_TP1_MULT)
    assert d["tp1_pct"] >= tp1_floor - 1e-6
    # geometry is no longer inverted: TP1 >= SL by construction
    assert d["tp1_pct"] >= d["sl_pct"] - 1e-6
    # fee-adjusted breakeven win rate dropped from the ZEC-era 71% to <= 62%
    # (net win 0.55% vs net loss 0.83% on this fixture; the live ZEC trade
    # needed 71.4%). The fee-survival rung rescues the give-back zone on
    # top of this, and TP2 (2.5x ATR) plus the ladder improve realized R.
    net_win = d["tp1_pct"] - 2.0 * settings.TRADING_FEE_PCT
    net_loss = d["sl_pct"] + 2.0 * settings.TRADING_FEE_PCT
    assert net_loss / (net_win + net_loss) <= 0.62


def test_double_ind_tp1_floor_tracks_fee_schedule(monkeypatch):
    """Doubling the fee schedule raises the effective TP1 floor."""
    from tests.test_v522_double_indicator import _burst_df
    from src.strategies.double_indicator_strategy import DoubleIndicatorStrategy
    monkeypatch.setattr(settings, "TRADING_FEE_PCT", 0.2)
    sig = DoubleIndicatorStrategy().analyze(_burst_df(), "MOMUSDT")
    assert sig.direction == "bullish"
    d = sig.details
    assert d["fee_rt_pct"] == pytest.approx(0.4)
    assert d["tp1_pct"] >= 0.4 * settings.DOUBLE_IND_FEE_TP1_MULT - 1e-6
