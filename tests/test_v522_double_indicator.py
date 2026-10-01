"""
v5.22 "Double Indicator" - the user's documented strategy, spot-only,
long-only (BB 11/3 + SuperTrend 2/2 on 1m candles).

The user's document specifies, verbatim:
  * Bollinger Bands period 11, deviation 3
  * SuperTrend ATR 2, multiplier 2
  * BUY: >= 3 consecutive green candles ABOVE the SuperTrend line and very
    close to the UPPER Bollinger band - "100% conditions or nothing"
  * upward side only ("عند الصعود فقط"), spot only

Covered here:
  - supertrend indicator: exact uptrend/downtrend behaviour
  - the binary checklist: fires ONLY on the full setup; 2 green candles,
    a bearish SuperTrend, or distance from the band each kill the signal
  - LONG-ONLY: the mirrored sell setup can never produce a bearish signal
  - forming-candle safety (signals are computed on CLOSED candles only)
  - fee-survival exit floors (SL/TP1/TP2 percentage floors)
  - scanner semi-stable/flat gates, ban guard, per-symbol burst cooldown
  - rec builder + risk plumbing: harmony exemption, channel caps
    (concurrent / spacing / BTC tide), ledger flag restore
  - session clock + regime router treat the channel as breakout-class
"""
import importlib
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from tests.test_v5_veteran import make_manager, make_position
import src.risk.manager as manager_module

cfgmod = importlib.import_module("config.settings")
_st = importlib.import_module("src.indicators.supertrend")
stratmod = importlib.import_module("src.strategies.double_indicator_strategy")
mommod = importlib.import_module("src.analysis.momentum_scanner")
anamod = importlib.import_module("src.analysis.analyzer")
scmod = importlib.import_module("src.analysis.session_clock")
DoubleIndicatorStrategy = stratmod.DoubleIndicatorStrategy


# ------------------------------------------------------------------
# Builders
# ------------------------------------------------------------------
def _burst_df(n_base=45, base_sigma=0.05, burst_steps=(1.2, 2.6, 4.2),
              base_price=100.0, base_vol=800.0, burst_vol=(2500., 3000., 3400.),
              with_close_time=False):
    """The pattern the doc targets: a quiet squeeze base, then a vertical
    3-candle green burst (each candle opens at the previous close)."""
    rng = np.random.default_rng(7)
    base = base_price + rng.normal(0, base_sigma, n_base)
    burst = base_price + np.array(burst_steps)
    closes = np.concatenate([base, burst])
    opens = np.concatenate([[closes[0]], closes[:-1]])
    vol = np.concatenate([np.full(n_base, base_vol), np.asarray(burst_vol)])
    idx = pd.date_range("2026-09-30", periods=len(closes), freq="min",
                        tz="UTC")
    df = pd.DataFrame({"open": opens, "close": closes, "volume": vol},
                      index=idx)
    df["high"] = np.maximum(opens, closes) * 1.0006
    df["low"] = np.minimum(opens, closes) * 0.9994
    if with_close_time:
        df["close_time"] = idx + pd.Timedelta(minutes=1)
    return df


def _down_burst_df(n_down=45):
    """The mirrored SELL setup from the doc (3 red candles below SuperTrend,
    near the LOWER band). The bot must NEVER trade it (spot, long-only)."""
    rng = np.random.default_rng(11)
    base = 100 + rng.normal(0, 0.05, n_down)
    burst = np.array([98.8, 97.4, 95.8])
    closes = np.concatenate([base, burst])
    opens = np.concatenate([[closes[0]], closes[:-1]])
    vol = np.concatenate([np.full(n_down, 800.0), [2600., 3100., 3500.]])
    idx = pd.date_range("2026-09-30", periods=len(closes), freq="min",
                        tz="UTC")
    df = pd.DataFrame({"open": opens, "close": closes, "volume": vol},
                      index=idx)
    df["high"] = np.maximum(opens, closes) * 1.0006
    df["low"] = np.minimum(opens, closes) * 0.9994
    return df


def _candidate(symbol="MOMUSDT", score=78.0, price=104.2, tp1=104.72,
               tp2=105.35, sl=103.63, rr=1.68, timeframe="1m"):
    """A momentum-scanner candidate as _scan_one returns it."""
    return {
        "symbol": symbol,
        "score": score,
        "direction": "bullish",
        "timeframe": timeframe,
        "current_price": price,
        "stop_loss": sl,
        "take_profit": tp1,
        "take_profit_2": tp2,
        "risk_reward_ratio": rr,
        "atr": 0.35,
        "atr_pct": 0.34,
        "pct_b_last": 0.91,
        "pct_b_avg": 0.95,
        "volume_ratio": 3.0,
        "st_line": 103.4,
        "signals": ["3 green candles above SuperTrend(2,2)",
                    "near upper BB(11,3)"],
        "green_run": 3,
        "analyzed_at": "2026-09-30T12:00:00+00:00",
    }


def _mom_rec(symbol="MOMUSDT"):
    c = _candidate(symbol=symbol)
    return anamod.build_double_ind_rec(c)


# ============================================
# SuperTrend indicator
# ============================================
def test_supertrend_uptrend_direction_and_line():
    idx = pd.date_range("2026-09-30", periods=80, freq="min", tz="UTC")
    close = pd.Series(np.linspace(100, 130, 80), index=idx)
    st, tr = _st.supertrend(close * 1.001, close * 0.999, close, 2, 2.0)
    assert tr.iloc[-3:].tolist() == [1, 1, 1]
    assert bool((st.iloc[-3:] < close.iloc[-3:]).all()), \
        "in an uptrend the ST line must sit below price"


def test_supertrend_downtrend_flip():
    idx = pd.date_range("2026-09-30", periods=80, freq="min", tz="UTC")
    close = pd.Series(np.linspace(130, 100, 80), index=idx)
    st, tr = _st.supertrend(close * 1.001, close * 0.999, close, 2, 2.0)
    assert tr.iloc[-3:].tolist() == [-1, -1, -1]
    assert bool((st.iloc[-3:] > close.iloc[-3:]).all()), \
        "in a downtrend the ST line must sit above price"


# ============================================
# The binary checklist (100% or nothing)
# ============================================
def test_perfect_squeeze_burst_fires_bullish():
    sig = DoubleIndicatorStrategy().analyze(_burst_df(), "MOMUSDT")
    assert sig.direction == "bullish"
    assert sig.strategy == "double_indicator"
    d = sig.details
    assert d["green_run"] == 3
    assert d["st_trend"] == 1
    assert d["pct_b_last"] >= settings.DOUBLE_IND_BB_PCTB_MIN
    # exact doc settings echoed back
    assert d["bb_period"] == 11 and d["bb_dev"] == 3.0
    assert d["st_period"] == 2 and d["st_mult"] == 2.0


def test_two_green_only_is_neutral():
    df = _burst_df()
    # first burst candle goes red (open above close)
    i = len(df) - 3
    df.iloc[i, df.columns.get_loc("close")] = df["open"].iloc[i] - 0.2
    df.iloc[i, df.columns.get_loc("high")] = df["open"].iloc[i] + 0.05
    df.iloc[i, df.columns.get_loc("low")] = df["close"].iloc[i] * 0.999
    sig = DoubleIndicatorStrategy().analyze(df, "MOMUSDT")
    assert sig.direction == "neutral"


def test_bounce_in_downtrend_is_neutral():
    """A green bounce while SuperTrend is still bearish fails the
    'candles above the line' condition - the doc's rule #2. ST(2,2) bands
    are tight, so the bounce must be small enough to stay under the line."""
    down = np.linspace(120, 100, 50)
    bounce = np.array([100.10, 100.22, 100.35])  # green, but under the line
    closes = np.concatenate([down, bounce])
    opens = np.concatenate([[closes[0]], closes[:-1]])
    idx = pd.date_range("2026-09-30", periods=len(closes), freq="min",
                        tz="UTC")
    df = pd.DataFrame({"open": opens, "close": closes,
                       "volume": np.full(len(closes), 1200.0)}, index=idx)
    df["high"] = np.maximum(opens, closes) * 1.0006
    df["low"] = np.minimum(opens, closes) * 0.9994
    sig = DoubleIndicatorStrategy().analyze(df, "MOMUSDT")
    assert sig.direction == "neutral"
    assert sig.details["st_trend"] == -1


def test_far_from_upper_band_is_neutral():
    """Wide bands (no squeeze) keep percent_B low even on a green run -
    the doc's rule #3 ('very close to the upper band') kills it."""
    df = _burst_df(base_sigma=0.9, burst_steps=(1.0, 2.0, 3.0))
    sig = DoubleIndicatorStrategy().analyze(df, "MOMUSDT")
    assert sig.direction == "neutral"
    assert sig.details["pct_b_last"] < settings.DOUBLE_IND_BB_PCTB_MIN


def test_mirror_sell_setup_never_bearish():
    """The doc's sell setup (3 red candles below SuperTrend near the lower
    band) must NEVER become a signal - spot has no shorts and the user
    asked for the upward side only."""
    strat = DoubleIndicatorStrategy()
    sig = strat.analyze(_down_burst_df(), "MOMUSDT")
    assert sig.direction == "neutral"
    # belt and braces: the class has no bearish emission path at all
    for _ in range(5):
        s = strat.analyze(_down_burst_df(), "MOMUSDT")
        assert s.direction != "bearish"


def test_forming_candle_is_dropped():
    """Binance returns the in-progress candle as the last row; a signal on
    a forming candle can still turn red - the 100% checklist is only
    knowable on CLOSED candles."""
    df = _burst_df(with_close_time=True)
    closed_sig = DoubleIndicatorStrategy().analyze(df, "MOMUSDT")
    # append a red forming candle whose close_time is genuinely in the
    # future (the 2026-09-30 synthetic clock is in the past by now)
    last = df.iloc[-1]
    idx = df.index[-1] + pd.Timedelta(minutes=1)
    forming = pd.DataFrame({
        "open": [last["close"]], "close": [last["close"] * 0.995],
        "volume": [5000.0], "high": [last["close"] * 1.001],
        "low": [last["close"] * 0.99],
        "close_time": [pd.Timestamp.now(tz="UTC") + pd.Timedelta(minutes=1)],
    }, index=[idx])
    df_open = pd.concat([df, forming])
    assert df_open["close_time"].iloc[-1] > pd.Timestamp.now(tz="UTC")
    open_sig = DoubleIndicatorStrategy().analyze(df_open, "MOMUSDT")
    assert open_sig.direction == closed_sig.direction


# ============================================
# Fee-survival exits + score discipline
# ============================================
def test_exit_floors_respected_on_quiet_atr():
    """On a quiet coin the ATR-based legs collapse below fees; the
    percentage floors must take over (v5.18 'TP inside fees' lesson)."""
    df = _burst_df(base_sigma=0.01, burst_steps=(0.35, 0.7, 1.1))
    sig = DoubleIndicatorStrategy().analyze(df, "MOMUSDT")
    assert sig.direction == "bullish"
    d = sig.details
    assert d["sl_pct"] >= settings.DOUBLE_IND_SL_MIN_PCT - 1e-6
    assert d["tp1_pct"] >= settings.DOUBLE_IND_TP1_MIN_PCT - 1e-6
    assert d["tp2_pct"] >= settings.DOUBLE_IND_TP2_MIN_PCT - 1e-6
    price = float(df["close"].iloc[-1])
    assert d["stop_loss"] < price < d["take_profit"] < d["take_profit_2"]
    rr = (d["take_profit_2"] - price) / max(price - d["stop_loss"], 1e-9)
    assert rr >= settings.MIN_RR_RATIO


def test_score_capped_at_channel_cap():
    sig = DoubleIndicatorStrategy().analyze(_burst_df(), "MOMUSDT")
    assert sig.score <= settings.DOUBLE_IND_CONF_CAP + 1e-9


def test_settings_v522_exist():
    for k in ("DOUBLE_IND_ENABLED", "DOUBLE_IND_TIMEFRAME",
              "DOUBLE_IND_TOP_N", "DOUBLE_IND_BB_PERIOD",
              "DOUBLE_IND_BB_DEV", "DOUBLE_IND_ST_PERIOD",
              "DOUBLE_IND_ST_MULT", "DOUBLE_IND_MIN_GREEN",
              "DOUBLE_IND_BB_PCTB_MIN", "DOUBLE_IND_MIN_ATR_PCT",
              "DOUBLE_IND_SL_MIN_PCT", "DOUBLE_IND_TP1_MIN_PCT",
              "DOUBLE_IND_TP2_MIN_PCT", "DOUBLE_IND_MAX_PER_CYCLE",
              "DOUBLE_IND_MAX_OPEN_CONCURRENT",
              "DOUBLE_IND_ENTRY_SPACING_MIN",
              "DOUBLE_IND_SYMBOL_COOLDOWN_MIN"):
        assert hasattr(settings, k), f"missing setting {k}"
    # the doc's exact indicator settings
    assert settings.DOUBLE_IND_BB_PERIOD == 11
    assert settings.DOUBLE_IND_BB_DEV == 3.0
    assert settings.DOUBLE_IND_ST_PERIOD == 2
    assert settings.DOUBLE_IND_ST_MULT == 2.0
    assert settings.DOUBLE_IND_MIN_GREEN == 3


# ============================================
# Scanner gates
# ============================================
def test_scanner_skips_flat_pinned_coin():
    sc = mommod.MomentumScanner()
    rng = np.random.default_rng(3)
    flat = pd.DataFrame(
        {"open": 100.0, "close": 100 + rng.normal(0, 0.003, 80),
         "volume": np.full(80, 100.0)},
        index=pd.date_range("2026-09-30", periods=80, freq="min", tz="UTC"))
    flat["high"] = flat["close"] * 1.0002
    flat["low"] = flat["close"] * 0.9998
    reason = sc._gate_candidate(flat, "FLATUSDT")
    assert reason and "semi-stable" in reason or "flat" in reason


class _FakeFetcher:
    """Per-symbol candle source: burst df only for the requested symbol,
    None for anything else (mirrors a real shortlist fetch)."""

    def __init__(self, df_by_symbol):
        self.df_by_symbol = df_by_symbol
        self.calls = []

    def get_candles(self, symbol, interval, limit=200):
        self.calls.append((symbol, interval, limit))
        df = self.df_by_symbol.get(symbol)
        return df.copy() if df is not None else None


def test_scanner_returns_candidate_and_arms_cooldown():
    fake = _FakeFetcher({"MOMUSDT": _burst_df()})
    monkey_fetch = fake
    sc = mommod.MomentumScanner()
    orig_fetch = mommod.data_fetcher
    mommod.data_fetcher = monkey_fetch
    try:
        out = sc.scan(symbols=["MOMUSDT", "CALMUSDT"], max_candidates=5)
        assert len(out) == 1
        c = out[0]
        assert c["symbol"] == "MOMUSDT"
        assert c["direction"] == "bullish"
        assert c["timeframe"] == settings.DOUBLE_IND_TIMEFRAME
        assert c["risk_reward_ratio"] >= settings.MIN_RR_RATIO
        assert fake.calls[0][1] == settings.DOUBLE_IND_TIMEFRAME
        assert fake.calls[0][2] == settings.DOUBLE_IND_KLINES_LIMIT
        # burst cooldown armed -> an immediate re-scan finds nothing
        out2 = sc.scan(symbols=["MOMUSDT"], max_candidates=5)
        assert out2 == []
    finally:
        mommod.data_fetcher = orig_fetch


def test_scanner_disabled_returns_empty():
    sc = mommod.MomentumScanner()
    orig = settings.DOUBLE_IND_ENABLED
    try:
        settings.DOUBLE_IND_ENABLED = False
        assert sc.scan(symbols=["MOMUSDT"]) == []
    finally:
        settings.DOUBLE_IND_ENABLED = orig


def test_scanner_ban_guard_returns_empty():
    rlmod = importlib.import_module("src.core.rate_limiter")
    sc = mommod.MomentumScanner()
    orig = rlmod.rate_limiter.cooldown_remaining
    try:
        rlmod.rate_limiter.cooldown_remaining = lambda: 300.0
        assert sc.scan(symbols=["MOMUSDT"]) == []
    finally:
        rlmod.rate_limiter.cooldown_remaining = orig


# ============================================
# Rec builder + risk plumbing
# ============================================
def test_build_double_ind_rec_fields():
    rec = _mom_rec()
    assert rec["symbol"] == "MOMUSDT"
    assert rec["direction"] == "bullish"  # LONG-ONLY channel
    assert rec["boosted_from_momentum"] is True
    assert rec["entry_type"] == "market"
    assert rec["timeframe"] == settings.DOUBLE_IND_TIMEFRAME
    assert rec["signals"][0]["strategy"] == "double_indicator"
    assert rec["confidence"] <= settings.DOUBLE_IND_CONF_CAP
    assert rec["admission_confidence"] == rec["confidence"]
    assert rec["expected_rise_pct"] >= settings.MIN_EXPECTED_RISE
    price, tp2 = rec["current_price"], rec["take_profit_2"]
    assert abs(tp2 - price) / price * 100.0 >= settings.DOUBLE_IND_TP2_MIN_PCT - 1e-6


def test_momentum_rec_passes_validation_without_harmony(rm):
    """The channel's admission is the binary checklist; the strategy-scale
    MIN_HARMONY gate must not silently veto it (same contract as the
    bottom channel)."""
    rec = _mom_rec()
    rec["harmony"] = 0.0  # prove the exemption, not the harmony value
    valid, reasons = rm.validate_recommendation(rec)
    assert valid, reasons


@pytest.fixture()
def rm(tmp_path, monkeypatch):
    """Isolated RiskManager with the momentum tide gate deterministically
    off (the tide tests enable it explicitly)."""
    monkeypatch.setattr(settings, "DOUBLE_IND_BTC_TIDE_GATE", False)
    return make_manager(tmp_path, monkeypatch)


def test_momentum_concurrent_cap_blocks_second(rm):
    for s in ("MOM1USDT",):
        p = make_position(symbol=s)
        p["boosted_from_momentum"] = True
        rm.open_positions.append(p)
    result = rm.open_paper_position(_mom_rec("NEWUSDT"))
    assert result["status"] == "rejected"
    assert any("concurrent cap" in r for r in result["reasons"])


def test_momentum_entry_spacing_blocks_rapid_refire(rm, monkeypatch):
    # the concurrent cap (1) must not mask the spacing check here
    monkeypatch.setattr(settings, "DOUBLE_IND_MAX_OPEN_CONCURRENT", 2)
    p = make_position(symbol="MOM1USDT")
    p["boosted_from_momentum"] = True
    p["entry_time"] = (datetime.now(timezone.utc)
                       - timedelta(minutes=5)).isoformat()
    rm.open_positions.append(p)
    result = rm.open_paper_position(_mom_rec("NEWUSDT"))
    assert result["status"] == "rejected"
    assert any("spacing" in r for r in result["reasons"])


def test_momentum_tide_gate_blocks_bearish_btc(rm, monkeypatch):
    monkeypatch.setattr(settings, "DOUBLE_IND_BTC_TIDE_GATE", True)
    rm._tide_snapshot = lambda: ("bearish", 0.0)
    result = rm.open_paper_position(_mom_rec("NEWUSDT"))
    assert result["status"] == "rejected"
    assert any("tide gate" in r for r in result["reasons"])


def test_position_from_row_restores_momentum_flag():
    row = {
        "symbol": "MOMUSDT", "direction": "bullish", "entry_price": 104.2,
        "stop_loss": 103.6, "take_profit": 104.7, "tp2": 105.3,
        "size": 0.09, "notional_usd": 10.0, "entry_fee": 0.01,
        "entry_time": "2026-09-30T12:00:00+00:00", "confidence": 78.0,
        "paper": 1, "trade_uid": "TRD-Y", "initial_sl": 103.6,
        "initial_tp": 104.7, "tp1_taken": 0, "realized_pnl": 0.0,
        "partial_count": 0, "peak_price": 104.2, "trough_price": 104.2,
        "mfe_pct": 0.0, "mae_pct": 0.0,
        "strategy": "double_indicator",
    }
    pos = manager_module.RiskManager._position_from_row(row)
    assert pos["boosted_from_momentum"] is True
    assert pos["boosted_from_bottom"] is False


def test_position_from_row_market_strategy_not_momentum():
    row = {"symbol": "X", "strategy": "trend_pullback", "entry_price": 1.0,
           "stop_loss": 0.99, "take_profit": 1.05}
    pos = manager_module.RiskManager._position_from_row(row)
    assert pos["boosted_from_momentum"] is False


# ============================================
# Channel registration (session clock + regime router)
# ============================================
def test_session_clock_treats_channel_as_breakout():
    """Saturday momentum block now applies to double_indicator too."""
    info = {"session": "ny", "hour": 16, "is_saturday": True,
            "weekday": 5, "is_weekend": True}
    blocked, why = scmod.entry_gate(info, "double_indicator", False)
    assert blocked and "سبت" in why


def test_regime_router_includes_channel():
    rrmod = importlib.import_module("src.analysis.regime_router")
    assert "double_indicator" in rrmod.BREAKOUT_STRATS
    assert "double_indicator" in rrmod.ALL_STRATS
