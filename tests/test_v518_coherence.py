"""
v5.18 entry/exit coherence + semi-stable purge tests (dashboard forensics).

Production evidence (2026-09-27, 3 closed trades, 0 wins):
  XAUTUSDT / CRCLBUSDT / TRXUSDT - all bullish, all bottom_scanner_boost,
  confidence exactly 72.0, all closed ~9 minutes after entry with
  "Structural exit: Ichimoku regime flipped bearish" - i.e. the bot bought
  4h bottoms (price under the cloud -> regime ALREADY bearish) and the very
  next analysis cycle read "still bearish" as "flipped bearish" and killed
  every trade with fees. XAUTUSDT is a gold token (semi-stable) - the exact
  coin class the user asked to exclude.

Fixes covered:
  - bottom_scanner: BOTTOM_MIN_ATR_PCT floor (semi-stable/dead coins),
    BOTTOM_MIN_TP_PCT fee floor, statistical-flatness skip
  - trend_pullback: pullback + reversal candle are now MANDATORY for a
    full signal (old code reached 63 pts without either -> 333/334
    bullish flood in a bearish regime)
  - risk manager: entry_ichimoku_regime snapshot + grace window ->
    "still bearish" no longer exits a bottom-fishing long; a genuine
    flip (entry bullish -> bearish) still exits; opposite-signal >= 55
    stays armed even in grace; legacy positions keep old behaviour
  - analyzer boost path: session entry_gate applies to boosted recs
"""
import importlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from src.analysis.analyzer import build_bottom_rec
from src.analysis.bottom_scanner import score_bottom_candidate
from src.risk.manager import RiskManager
import src.risk.manager as manager_module

cfgmod = importlib.import_module("config.settings")


def _mod(name: str):
    return importlib.import_module(name)


def _utc(y, m, d, h, minute=0):
    return datetime(y, m, d, h, minute, tzinfo=timezone.utc)


def make_manager(tmp_path, monkeypatch):
    """RiskManager isolated from real data files (pattern of test_v5)."""
    from tests.test_v5_veteran import make_manager as _base
    return _base(tmp_path, monkeypatch)


def _candidate(score=65.0, n_signals=3, symbol="BOTTOMUSDT", atr_pct=1.2):
    """A bottom-scanner candidate as score_bottom_candidate returns it."""
    return {
        "symbol": symbol,
        "score": float(score),
        "current_price": 1.0,
        "recent_low": 0.98,
        "recent_high": 1.30,
        "distance_from_low_pct": 2.0,
        "position_in_range_pct": 6.7,
        "rsi": 28.0,
        "atr_pct": atr_pct,
        "stop_loss": 1.0 - 1.2 * (atr_pct / 100.0),
        "take_profit": 1.0 + 2.5 * (atr_pct / 100.0),
        "risk_reward_ratio": 2.083,
        "signals": [f"layer{i}" for i in range(n_signals)],
        "bb_percent_b": 0.03,
        "last_candle_bullish": True,
        "patterns_detected": ["hammer"],
        "analyzed_at": "2026-09-25T00:00:00+00:00",
    }


def _df(closes, opens=None, volumes=None, freq="4h"):
    n = len(closes)
    closes = np.asarray(closes, dtype=float)
    if opens is None:
        opens = np.concatenate([[closes[0]], closes[:-1]])
    opens = np.asarray(opens, dtype=float)
    highs = np.maximum(opens, closes) * 1.0005
    lows = np.minimum(opens, closes) * 0.9995
    vol = np.full(n, 1000.0) if volumes is None else np.asarray(volumes, float)
    idx = pd.date_range("2026-09-20", periods=n, freq=freq, tz="UTC")
    return pd.DataFrame({"open": opens, "high": highs, "low": lows,
                         "close": closes, "volume": vol}, index=idx)


# ======================================================================
# Settings defaults
# ======================================================================
def test_settings_v518_defaults():
    assert settings.BOTTOM_MIN_ATR_PCT == pytest.approx(0.60)
    assert settings.BOTTOM_MIN_TP_PCT == pytest.approx(0.90)
    assert settings.STRUCTURAL_EXIT_GRACE_MIN == pytest.approx(30.0)
    assert settings.ATR_PCT_MIN == pytest.approx(0.30)


# ======================================================================
# Bottom scanner: semi-stable / fee gates
# ======================================================================
def test_scanner_rejects_semi_stable_low_atr():
    """A gold-token-like series (XAUT): near-fixed price, ATR% ~0.07."""
    base = 1.0
    closes = [base * (1 + (0.0004 if i % 2 else -0.0004)) for i in range(60)]
    r = score_bottom_candidate("XAUTUSDT", _df(closes))
    assert r.get("skip"), f"semi-stable must be skipped, got {r}"
    assert "Semi-stable" in r.get("reason", "") or "ATR" in r.get("reason", "")


def test_scanner_accepts_genuinely_volatile_bottom():
    """A real dump-then-bounce alt: ATR% ~1.5 passes every v5.18 gate."""
    closes = []
    p = 1.50
    for i in range(58):          # steady dump ~1.5% per bar
        closes.append(p)
        p *= 0.985
    closes += [p * 1.005, p * 1.010, p * 1.016]  # small bounce, bullish close
    r = score_bottom_candidate("DUMPUSDT", _df(closes))
    assert not r.get("skip"), f"volatile bottom must pass, got skip: {r.get('reason')}"
    assert r["atr_pct"] >= settings.BOTTOM_MIN_ATR_PCT
    assert r["risk_reward_ratio"] > 2.0  # 2.5/1.2 unchanged


def test_scanner_tp_fee_floor():
    """Even with ATR above the floor, a TP inside fees is rejected."""
    # atr_pct just above 0.6 -> tp = 2.5*0.65 = 1.63% ... use 0.36 to force
    # the ATR gate; for a pure TP-gate hit we monkeypatch the floor instead.
    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(cfgmod.settings, "BOTTOM_MIN_ATR_PCT", 0.10)
        closes = [1.0 * (1 + (0.0012 if i % 2 else -0.0012)) for i in range(60)]
        r = score_bottom_candidate("THINUSDT", _df(closes))
        # ATR ~0.12% >= 0.10 floor, but TP = 2.5*0.12 = 0.30% < 0.90 floor
        assert r.get("skip"), f"fee-food must be skipped, got {r}"
        assert "TP inside fees" in r.get("reason", "")
    finally:
        monkey.undo()


# ======================================================================
# trend_pullback: pullback + candle mandatory
# ======================================================================
def _uptrend_df(with_pullback=True, hammer=True, n=90):
    base = list(np.linspace(100.0, 130.0, n - 3))
    if with_pullback:
        tail = [129.0, 128.2, 129.4]   # dip then recover
    else:
        tail = [132.0, 133.5, 135.0]   # accelerating away from EMA21
    closes = base + tail
    opens = [c - 0.15 for c in closes]
    if with_pullback and hammer:
        # last bar: hammer (long lower wick, small bullish body)
        opens[-1] = 129.0
        closes[-1] = 129.5
    elif with_pullback and not hammer:
        # last bar: plain bearish candle (no reversal pattern)
        opens[-1] = 129.3
        closes[-1] = 128.6
    df = _df(closes, opens=opens)
    if with_pullback:
        from src.indicators.technical import ema
        e21 = float(ema(df["close"], 21).iloc[-1])
        # bar -2 dips to touch EMA21
        df.loc[df.index[-2], "low"] = e21 * 0.995
        if hammer:
            df.loc[df.index[-1], "low"] = e21 * 0.992  # hammer wick
    return df


def test_pullback_no_pullback_is_neutral_not_bull():
    """Uptrend accelerating away from EMA21: OLD code fired full bull
    (63 pts from trend+ADX+volume+MACD) - a pure chase. Must be neutral."""
    strat = _mod("src.strategies.trend_pullback_strategy"). \
        TrendPullbackStrategy()
    sig = strat.analyze(_uptrend_df(with_pullback=False), "TESTUSDT")
    assert sig.direction == "neutral", sig
    assert any("No pullback" in r for r in sig.reasons)


def test_pullback_with_candle_is_full_bull():
    strat = _mod("src.strategies.trend_pullback_strategy"). \
        TrendPullbackStrategy()
    sig = strat.analyze(_uptrend_df(with_pullback=True, hammer=True),
                        "TESTUSDT")
    assert sig.direction == "bullish", sig
    assert sig.score >= 50, sig


def test_pullback_without_candle_is_partial_only():
    strat = _mod("src.strategies.trend_pullback_strategy"). \
        TrendPullbackStrategy()
    sig = strat.analyze(_uptrend_df(with_pullback=True, hammer=False),
                        "TESTUSDT")
    assert sig.direction == "bullish", sig
    assert any("No reversal candle" in r for r in sig.reasons)
    # partial = raw score halved -> cannot carry full-strength confidence
    assert sig.confidence < 0.60, sig


# ======================================================================
# Structural exit: entry-regime coherence + grace window
# ======================================================================
_BEAR_ICHO = {"regime": "bearish", "kijun": 90.0, "tenkan": 95.0,
              "price_vs_kijun": "below", "tk_cross_recent": "bearish",
              "tk_state": "bearish"}
_BULL_ICHO = {"regime": "bullish", "kijun": 110.0, "tenkan": 105.0,
              "price_vs_kijun": "above", "tk_cross_recent": "bullish",
              "tk_state": "bullish"}


def _pos(direction="bullish", entry_regime="bearish", age_min=120):
    return {
        "symbol": "TESTUSDT",
        "direction": direction,
        "entry_time": (datetime.now(timezone.utc)
                       - timedelta(minutes=age_min)).isoformat(),
        "entry_ichimoku_regime": entry_regime,
        "stop_loss": 90.0 if direction == "bullish" else 110.0,
    }


def test_still_bearish_after_bearish_entry_no_exit(tmp_path, monkeypatch):
    rm = make_manager(tmp_path, monkeypatch)
    action, reason = rm.evaluate_structural_exit(
        _pos(entry_regime="bearish"),
        {"direction": "neutral", "confidence": 0, "ichimoku": _BEAR_ICHO},
        100.0)
    assert action == "none", (action, reason)


def test_genuine_flip_after_bullish_entry_exits(tmp_path, monkeypatch):
    rm = make_manager(tmp_path, monkeypatch)
    action, reason = rm.evaluate_structural_exit(
        _pos(entry_regime="bullish"),
        {"direction": "neutral", "confidence": 0, "ichimoku": _BEAR_ICHO},
        100.0)
    assert action == "exit"
    assert reason == "Ichimoku regime flipped bearish"


def test_legacy_position_without_snapshot_keeps_old_behaviour(tmp_path,
                                                              monkeypatch):
    rm = make_manager(tmp_path, monkeypatch)
    pos = _pos(entry_regime="bearish")
    pos.pop("entry_ichimoku_regime")
    action, _ = rm.evaluate_structural_exit(
        pos,
        {"direction": "neutral", "confidence": 0, "ichimoku": _BEAR_ICHO},
        100.0)
    assert action == "exit"


def test_grace_window_suppresses_regime_exit(tmp_path, monkeypatch):
    rm = make_manager(tmp_path, monkeypatch)
    action, _ = rm.evaluate_structural_exit(
        _pos(entry_regime="bullish", age_min=5),
        {"direction": "neutral", "confidence": 0, "ichimoku": _BEAR_ICHO},
        100.0)
    assert action == "none"


def test_opposite_signal_still_exits_during_grace(tmp_path, monkeypatch):
    """User rule: opposite analysis >= 55 conf closes NOW - grace never
    suppresses it (only regime/Kijun/Tenkan reactions are deferred)."""
    rm = make_manager(tmp_path, monkeypatch)
    action, reason = rm.evaluate_structural_exit(
        _pos(entry_regime="bullish", age_min=5),
        {"direction": "bearish", "confidence": 62, "ichimoku": _BULL_ICHO},
        100.0)
    assert action == "exit"
    assert "Opposite" in reason


def test_mirror_still_bullish_after_bullish_entry_no_exit(tmp_path,
                                                         monkeypatch):
    """Short opened under a bullish regime is not killed by 'still bullish'."""
    rm = make_manager(tmp_path, monkeypatch)
    action, _ = rm.evaluate_structural_exit(
        _pos(direction="bearish", entry_regime="bullish"),
        {"direction": "neutral", "confidence": 0, "ichimoku": _BULL_ICHO},
        100.0)
    assert action == "none"


def test_mirror_flip_after_bearish_entry_exits(tmp_path, monkeypatch):
    rm = make_manager(tmp_path, monkeypatch)
    action, reason = rm.evaluate_structural_exit(
        _pos(direction="bearish", entry_regime="bearish"),
        {"direction": "neutral", "confidence": 0, "ichimoku": _BULL_ICHO},
        100.0)
    assert action == "exit"
    assert reason == "Ichimoku regime flipped bullish"


# ======================================================================
# Position snapshot at open time
# ======================================================================
def test_open_position_stores_entry_regime_snapshot(tmp_path, monkeypatch):
    rm = make_manager(tmp_path, monkeypatch)
    rec = build_bottom_rec(_candidate())
    rec["ichimoku"] = {"regime": "bearish"}
    result = rm.open_position(rec)
    assert result["status"] == "opened", result.get("reasons")
    pos = rm.open_positions[0]
    assert pos["entry_ichimoku_regime"] == "bearish"
    assert pos["boosted_from_bottom"] is True


def test_open_position_snapshot_none_when_no_ichimoku(tmp_path, monkeypatch):
    rm = make_manager(tmp_path, monkeypatch)
    rec = build_bottom_rec(_candidate())  # no ichimoku key at all
    result = rm.open_position(rec)
    assert result["status"] == "opened", result.get("reasons")
    pos = rm.open_positions[0]
    assert pos["entry_ichimoku_regime"] is None
    assert pos["boosted_from_bottom"] is True


# ======================================================================
# Session gate reaches the boost channel
# ======================================================================
def test_entry_gate_blocks_bottom_boost_in_chop_window():
    sc = _mod("src.analysis.session_clock")
    blocked, reason = sc.entry_gate(
        sc.session_info(_utc(2026, 9, 23, 12)),  # Wednesday 12:00 UTC
        "bottom_scanner_boost", False)
    assert blocked
    assert "نافذة الاختلاق" in reason


def test_entry_gate_allows_bottom_boost_in_london():
    sc = _mod("src.analysis.session_clock")
    blocked, _ = sc.entry_gate(
        sc.session_info(_utc(2026, 9, 23, 8)),  # Wednesday 08:00 UTC
        "bottom_scanner_boost", False)
    assert not blocked
