"""v5.25 - Micro-Scalp channel: the user's strategy VERBATIM on true
15s/30s candles (resampled from 1s klines), long-only, FIXED 1-minute hold.

Covers:
  * 1s -> 15s/30s resample math (OHLCV aggregation, bucket close_time,
    forming-bucket drop),
  * the doc's timeframe rule (15s calm / 30s very fast),
  * the 100% checklist (green run / above SuperTrend / near upper band),
  * LONG-ONLY by construction,
  * scanner plumbing (ban guard, RateLimitError abort, per-symbol cooldown),
  * rec contract (boosted_from_scalp, RR >= 1.5 by construction,
    admission through validate_recommendation),
  * risk-manager integration (channel cap, fixed 60s time exit, structural
    immunity, trailing skip, DB-restore flag).
"""
import importlib
from datetime import timedelta

import numpy as np
import pandas as pd

cfgmod = importlib.import_module("config.settings")
settings = cfgmod.settings
stratmod = importlib.import_module("src.strategies.micro_scalp_strategy")
MicroScalpStrategy = stratmod.MicroScalpStrategy
resample_1s = stratmod.resample_1s
choose_timeframe = stratmod.choose_timeframe
scanmod = importlib.import_module("src.analysis.scalp_scanner")
ScalpScanner = scanmod.ScalpScanner
build_scalp_rec = scanmod.build_scalp_rec
from src.core.rate_limiter import RateLimitError
from src.utils.helpers import now_utc

from tests.test_v5_veteran import make_manager, make_position


# ------------------------------------------------------------------
# 1s kline builders
# ------------------------------------------------------------------
def build_1s(seconds=1000, flat=100.0, slope=0.0, slope_last=0,
             aligned_start=False):
    """One-second bars: flat base, optional linear slope over the last
    `slope_last` seconds. Aligned mode guarantees full 15s buckets."""
    if aligned_start:
        # 30s floor: aligned to BOTH the 15s and 30s bucket grids
        start = pd.Timestamp.now(tz="UTC").floor("30s") - \
            pd.Timedelta(seconds=seconds)
        idx = pd.date_range(start=start, periods=seconds, freq="s",
                            tz="UTC")
    else:
        end = now_utc().replace(microsecond=0)
        idx = pd.date_range(end=end, periods=seconds, freq="s", tz="UTC")
    close = np.full(seconds, flat)
    if slope_last > 0:
        close[-slope_last:] = flat + slope * np.arange(1, slope_last + 1)
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + 0.0001
    low = np.minimum(open_, close) - 0.0001
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": np.full(seconds, 10.0),
         "close_time": idx + pd.Timedelta(seconds=1)},
        index=idx)


RAMP = dict(slope=0.0015, slope_last=1000)  # passes every checklist gate


def build_1s_gap_burst(seconds=1000, flat=100.0,
                       steps=(0.2, 0.4, 0.8, 1.6, 3.2), aligned_start=True):
    """v5.27: a fee-SURVIVABLE market fixture. The old RAMP (0.015% per
    15s candle) is exactly the fee-food the live day proved untradeable:
    its whole 60s hold moved ~0.06% against a 0.2% round-trip fee. This
    builder keeps the base flat and prints a parabolic gap-spike over the
    last len(steps) closed 15s buckets - the only realistic shape that
    passes BOTH the verbatim near-band rule (BB 11/3) and the fee-linked
    ATR floor (max(0.25%, fees x 1.25))."""
    if aligned_start:
        start = pd.Timestamp.now(tz="UTC").floor("30s") - \
            pd.Timedelta(seconds=seconds)
        idx = pd.date_range(start=start, periods=seconds, freq="s",
                            tz="UTC")
    else:
        end = now_utc().replace(microsecond=0)
        idx = pd.date_range(end=end, periods=seconds, freq="s", tz="UTC")
    close = np.full(seconds, flat)
    n = len(steps)
    for k in range(1, n + 1):
        # the k-th burst bucket from the end: its first 1s bar carries the gap
        close[seconds - k * 15:] += steps[n - k]
    open_ = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum(open_, close) + 0.0001
    low = np.minimum(open_, close) - 0.0001
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": np.full(seconds, 10.0),
         "close_time": idx + pd.Timedelta(seconds=1)},
        index=idx)


# ------------------------------------------------------------------
# Resample math
# ------------------------------------------------------------------
def test_resample_15s_math():
    tiny = build_1s(120, slope=0.01, slope_last=120, aligned_start=True)
    r = resample_1s(tiny, 15)
    assert len(r) == 8
    b0 = r.iloc[0]
    assert abs(float(b0["open"]) - float(tiny["open"].iloc[0])) < 1e-9
    assert abs(float(b0["high"]) - max(tiny["high"].iloc[:15])) < 1e-9
    assert abs(float(b0["low"]) - min(tiny["low"].iloc[:15])) < 1e-9
    assert abs(float(b0["close"]) - float(tiny["close"].iloc[14])) < 1e-9
    assert abs(float(b0["volume"]) - 150.0) < 1e-9  # 15 bars x 10.0


def test_resample_30s_buckets_and_close_time():
    tiny = build_1s(120, slope=0.01, slope_last=120, aligned_start=True)
    r = resample_1s(tiny, 30)
    assert len(r) == 4
    # close_time = bucket end (exclusive); forming detection relies on it
    assert (r["close_time"] > r.index).all()
    assert (r["close_time"] - r.index == pd.Timedelta(seconds=30)).all()


def test_forming_bucket_dropped():
    # ends exactly at "now": the last 15s bucket is still forming
    df = build_1s(1000, **RAMP)
    r = resample_1s(df, 15)
    from src.strategies.double_indicator_strategy import last_closed_bars
    closed = last_closed_bars(r)
    assert len(closed) == len(r) - 1


def test_1s_interval_is_accepted():
    from src.core.data_fetcher import DataFetcher
    assert "1s" in DataFetcher.INTERVALS


# ------------------------------------------------------------------
# Timeframe rule (the doc: 15s calm / 30s very fast)
# ------------------------------------------------------------------
def test_tf_calm_15s():
    assert choose_timeframe(build_1s(1000, slope=0.0001, slope_last=1000)) \
        == 15


def test_tf_fast_30s():
    assert choose_timeframe(build_1s(1000, slope=0.006, slope_last=1000)) \
        == 30


# ------------------------------------------------------------------
# The 100% checklist
# ------------------------------------------------------------------
def test_checklist_fires_on_fee_survivable_burst(monkeypatch):
    monkeypatch.setattr(settings, "SCALP_BB_NEAR_PCT", 0.12)
    sig = MicroScalpStrategy().analyze(build_1s_gap_burst(), "TESTUSDT")
    assert sig.direction == "bullish", sig.reasons
    d = sig.details
    assert d["green_run"] >= 3
    assert d["st_trend"] == 1
    assert d["timeframe"] in ("15s", "30s")
    # spot-adaptation ladder: disaster-SL floor and TP2 floor respected
    assert d["sl_pct"] >= settings.SCALP_SL_MIN_PCT - 0.01
    assert d["tp2_pct"] >= settings.SCALP_TP2_MIN_PCT - 0.01
    # v5.27: TP1 must clear round-trip fees plus the net minimum
    assert d["tp1_pct"] >= d["fee_rt_pct"] + settings.SCALP_MIN_NET_MOVE_PCT
    # and the fixture itself must be fee-honest (ATR above the floor)
    assert d["atr_pct"] >= d["min_atr_floor_pct"]


def test_checklist_requires_green_run(monkeypatch):
    monkeypatch.setattr(settings, "SCALP_BB_NEAR_PCT", 99.0)
    # checklist unit test: fee floors OFF so the failure is the checklist's
    _fee_floors_off(monkeypatch)
    df = build_1s(1000, **RAMP)
    # turn the LAST CLOSED 15s bucket red (its final 1s bar closes red) -
    # the current bucket is still forming and gets dropped
    from src.strategies.double_indicator_strategy import last_closed_bars
    closed = last_closed_bars(resample_1s(df, 15))
    bucket_start = closed.index[-1]
    mask = (df.index >= bucket_start) & (
        df.index < bucket_start + pd.Timedelta(seconds=15))
    # dip deeper than the intra-bucket ramp step (~0.033) so the bucket
    # itself closes red
    df.loc[mask, "close"] = df.loc[mask, "open"] - 0.05
    sig = MicroScalpStrategy().analyze(df, "TESTUSDT")
    assert sig.direction == "neutral"
    assert any("أخضر" in r or "green" in r.lower() for r in sig.reasons)


def test_checklist_requires_above_supertrend(monkeypatch):
    monkeypatch.setattr(settings, "SCALP_BB_NEAR_PCT", 99.0)
    _fee_floors_off(monkeypatch)
    df = build_1s(1000, **RAMP)
    # crash the tail through the SuperTrend line (red candles below it)
    df.iloc[-30:, df.columns.get_loc("close")] *= 0.99
    df.iloc[-30:, df.columns.get_loc("high")] *= 0.99
    df.iloc[-30:, df.columns.get_loc("low")] *= 0.99
    sig = MicroScalpStrategy().analyze(df, "TESTUSDT")
    assert sig.direction == "neutral"
    assert any("SuperTrend" in r for r in sig.reasons)


def test_checklist_requires_near_band(monkeypatch):
    # 0.05% nearness vs a ramp whose band distance is ~0.10% -> far
    monkeypatch.setattr(settings, "SCALP_BB_NEAR_PCT", 0.05)
    _fee_floors_off(monkeypatch)
    sig = MicroScalpStrategy().analyze(build_1s(1000, **RAMP), "TESTUSDT")
    assert sig.direction == "neutral"
    assert any("الحد العلوي" in r for r in sig.reasons)


def _fee_floors_off(monkeypatch):
    """Checklist unit tests isolate the checklist from the v5.27 fee
    floors (RAMP's ATR is below the fee-linked floor by design)."""
    monkeypatch.setattr(settings, "SCALP_MIN_ATR_PCT", 0.0)
    monkeypatch.setattr(settings, "SCALP_FEE_COVER_MULT", 0.0)


def test_fee_linked_atr_floor_blocks_pinned_market(monkeypatch):
    """v5.27 THE live-day lesson: a market whose resampled-TF ATR cannot
    cover the round-trip fee is untradeable on a 60s time exit no matter
    how clean the checklist looks (6/6 fee losses on day one)."""
    assert settings.SCALP_FEE_COVER_MULT > 0
    sig = MicroScalpStrategy().analyze(build_1s(1000, **RAMP), "TESTUSDT")
    assert sig.direction == "neutral"
    assert any("fee-food" in r or "fee-linked" in r for r in sig.reasons)


def test_fee_floor_scales_with_trading_fee(monkeypatch):
    """The floor is derived from TRADING_FEE_PCT - doubling the fee
    schedule doubles the minimum tradeable volatility."""
    monkeypatch.setattr(settings, "TRADING_FEE_PCT", 0.2)  # 0.4% RT -> 0.5% floor
    sig = MicroScalpStrategy().analyze(build_1s_gap_burst(), "TESTUSDT")
    assert sig.direction == "neutral"
    assert any("fee-linked" in r for r in sig.reasons)


def test_long_only_by_construction(monkeypatch):
    """A falling market can never produce a signal - there is no bearish
    branch in the strategy at all (spot, 'عند الصعود فقط')."""
    _fee_floors_off(monkeypatch)
    sig = MicroScalpStrategy().analyze(
        build_1s(1000, slope=-0.002, slope_last=1000), "TESTUSDT")
    assert sig.direction == "neutral"
    assert sig.direction != "bearish"


def test_insufficient_data_is_neutral():
    assert MicroScalpStrategy().analyze(
        build_1s(120, slope=0.01, slope_last=120), "TESTUSDT").direction \
        == "neutral"
    assert MicroScalpStrategy().analyze(None, "TESTUSDT").direction == \
        "neutral"


# ------------------------------------------------------------------
# Scanner plumbing
# ------------------------------------------------------------------
def _fake_fetcher(monkeypatch, df_by_symbol):
    calls = {"n": 0}

    class _FakeFetcher:
        def get_candles(self, symbol, interval="1h", limit=200):
            calls["n"] += 1
            calls["interval"] = interval
            out = df_by_symbol.get(symbol)
            if isinstance(out, Exception):
                raise out
            return out

    monkeypatch.setattr(scanmod, "data_fetcher", _FakeFetcher())
    return calls


def test_scanner_produces_candidate(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "SCALP_BB_NEAR_PCT", 0.12)
    df = build_1s_gap_burst()
    calls = _fake_fetcher(monkeypatch, {"TESTUSDT": df})
    scanner = ScalpScanner()
    cands = scanner.scan(symbols=["TESTUSDT"])
    assert calls["n"] == 1
    assert calls["interval"] == "1s"
    assert len(cands) == 1
    c = cands[0]
    assert c["symbol"] == "TESTUSDT"
    assert c["direction"] == "bullish"
    assert c["risk_reward_ratio"] >= 1.5
    assert c["timeframe"] in ("15s", "30s")
    assert c["green_run"] >= 3
    # fired -> per-symbol cooldown engaged
    assert scanner._symbol_on_cooldown("TESTUSDT")


def test_scanner_ban_guard(monkeypatch):
    from src.core.rate_limiter import rate_limiter
    monkeypatch.setattr(rate_limiter, "cooldown_remaining", lambda: 300.0)
    scanner = ScalpScanner()
    calls = _fake_fetcher(monkeypatch, {"TESTUSDT": build_1s(1000, **RAMP)})
    assert scanner.scan(symbols=["TESTUSDT"]) == []
    assert calls["n"] == 0  # zero REST weight during a ban


def test_scanner_rate_limit_aborts_tick(monkeypatch):
    df = build_1s(1000, **RAMP)
    calls = _fake_fetcher(monkeypatch, {
        "AAAUSDT": RateLimitError("rate budget refused"),
        "BBBUSDT": df,
    })
    scanner = ScalpScanner()
    assert scanner.scan(symbols=["AAAUSDT", "BBBUSDT"]) == []
    assert calls["n"] == 1  # aborted after the first refusal


def test_scanner_disabled_returns_empty(monkeypatch):
    monkeypatch.setattr(settings, "SCALP_ENABLED", False)
    scanner = ScalpScanner()
    calls = _fake_fetcher(monkeypatch, {"TESTUSDT": build_1s(1000, **RAMP)})
    assert scanner.scan(symbols=["TESTUSDT"]) == []
    assert calls["n"] == 0


# ------------------------------------------------------------------
# Rec contract + admission
# ------------------------------------------------------------------
def _candidate():
    return {
        "symbol": "TESTUSDT", "score": 72.0, "direction": "bullish",
        "timeframe": "15s", "current_price": 100.0, "stop_loss": 99.2,
        "take_profit": 100.35, "take_profit_2": 101.6,
        "risk_reward_ratio": 2.0, "atr": 0.05, "atr_pct": 0.05,
        "near_upper_dist_pct": 0.08, "volume_ratio": 1.0,
        "signals": ["3 شموع 15ث خضراء", "قريبة من الحد العلوي"],
        "green_run": 3, "analyzed_at": now_utc().isoformat(),
    }


def test_build_scalp_rec_contract():
    rec = build_scalp_rec(_candidate())
    assert rec["boosted_from_scalp"] is True
    assert rec["direction"] == "bullish"  # LONG-ONLY
    assert rec["signals"][0]["strategy"] == "micro_scalp"
    assert rec["risk_reward_ratio"] >= 1.5
    assert rec["expected_rise_pct"] >= settings.MIN_EXPECTED_RISE
    assert rec["entry_type"] == "market"


def test_scalp_rec_passes_validation_and_opens(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "SCALP_BTC_TIDE_GATE", False)
    mgr = make_manager(tmp_path, monkeypatch)
    rec = build_scalp_rec(_candidate())
    valid, reasons = mgr.validate_recommendation(rec)
    assert valid, reasons  # harmony-exempt via boosted_from_scalp
    res = mgr.open_position(rec)
    assert res["status"] == "opened", res.get("reasons")
    pos = res["position"]
    assert pos["boosted_from_scalp"] is True
    assert pos["strategy"] == "micro_scalp"


def test_scalp_channel_cap_blocks_second(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "SCALP_BTC_TIDE_GATE", False)
    mgr = make_manager(tmp_path, monkeypatch)
    first = mgr.open_position(build_scalp_rec(_candidate()))
    assert first["status"] == "opened"
    second = build_scalp_rec(_candidate())
    second["symbol"] = "OTHERUSDT"
    res = mgr.open_position(second)
    assert res["status"] == "rejected"
    assert any("scalp concurrent cap" in r for r in res["reasons"])


# ------------------------------------------------------------------
# Risk-manager integration
# ------------------------------------------------------------------
def test_time_exit_closes_scalp_at_60s(tmp_path, monkeypatch):
    mgr = make_manager(tmp_path, monkeypatch)
    pos = make_position(entry=100.0, sl=99.2, tp=100.35, tp2=101.6)
    pos["boosted_from_scalp"] = True
    pos["strategy"] = "micro_scalp"
    pos["entry_time"] = (now_utc() - timedelta(seconds=90)).isoformat()
    mgr.open_positions = [pos]
    results = mgr.check_open_positions({"TESTUSDT": 100.5})
    assert len(results) == 1
    assert "Scalp time exit" in results[0].get("reason", "")
    assert len(mgr.open_positions) == 0


def test_time_exit_holds_inside_60s_window(tmp_path, monkeypatch):
    mgr = make_manager(tmp_path, monkeypatch)
    pos = make_position(entry=100.0, sl=99.2, tp=100.35, tp2=101.6)
    pos["boosted_from_scalp"] = True
    pos["entry_time"] = (now_utc() - timedelta(seconds=30)).isoformat()
    mgr.open_positions = [pos]
    results = mgr.check_open_positions({"TESTUSDT": 100.2})
    assert results == []
    assert len(mgr.open_positions) == 1


def test_disaster_sl_still_armed_for_scalp(tmp_path, monkeypatch):
    """The doc's expiry is the exit - but a catastrophic dump still hits
    the hard SL first (spot adaptation, checked BEFORE the time stop)."""
    mgr = make_manager(tmp_path, monkeypatch)
    pos = make_position(entry=100.0, sl=99.2, tp=100.35, tp2=101.6)
    pos["boosted_from_scalp"] = True
    pos["entry_time"] = (now_utc() - timedelta(seconds=20)).isoformat()
    mgr.open_positions = [pos]
    results = mgr.check_open_positions({"TESTUSDT": 99.0})
    assert len(results) == 1
    assert "Stop Loss" in results[0].get("reason", "")


def test_structural_exit_immune(tmp_path, monkeypatch):
    mgr = make_manager(tmp_path, monkeypatch)
    pos = make_position(entry=100.0, sl=99.2, tp=100.35)
    pos["boosted_from_scalp"] = True
    bearish_sig = {"direction": "bearish", "confidence": 80.0,
                   "ichimoku": {"regime": "bearish"}}
    action, reason = mgr.evaluate_structural_exit(pos, bearish_sig, 99.0)
    assert action == "none"
    assert reason is None


def test_trailing_skips_scalp(tmp_path, monkeypatch):
    mgr = make_manager(tmp_path, monkeypatch)
    pos = make_position(entry=100.0, sl=99.2, tp=110.0)
    pos["boosted_from_scalp"] = True
    mgr.open_positions = [pos]
    updates = mgr.apply_trailing_logic({"TESTUSDT": 105.0})  # +5% profit
    assert updates == []
    assert mgr.open_positions[0]["stop_loss"] == 99.2  # untouched


def test_db_restore_keeps_scalp_flag():
    from src.risk.manager import RiskManager
    row = {
        "symbol": "TESTUSDT", "direction": "bullish",
        "entry_price": 100.0, "stop_loss": 99.2, "take_profit": 100.35,
        "take_profit_2": 101.6, "notional_usd": 10.0, "size": 0.1,
        "entry_fee": 0.01, "entry_time": now_utc().isoformat(),
        "confidence": 72.0, "paper": 1, "trade_uid": "TRD-TEST-1",
        "strategy": "micro_scalp", "status": "open",
    }
    pos = RiskManager._position_from_row(row)
    assert pos["boosted_from_scalp"] is True
    row["strategy"] = "trend_pullback"
    assert RiskManager._position_from_row(row)["boosted_from_scalp"] is False
