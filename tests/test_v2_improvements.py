"""
Tests for v2 improvements:
  - ADX correctness (Wilder's algorithm)
  - New confidence model (strength x confluence)
  - Structure-aware SL/TP
  - Trailing stop ladder (direct jump)
  - Duplicate position guard
  - Telegram cooldown
Run: pytest tests/test_v2_improvements.py -v
"""
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.indicators import technical as ta
from src.utils.helpers import now_utc


def _sample_ohlcv(n=300, seed=42, trend=0.0):
    """Generate synthetic OHLCV data (optionally with a drift/trend)."""
    np.random.seed(seed)
    base = 100
    returns = np.random.normal(0.0005 + trend, 0.02, n)
    closes = base * np.exp(np.cumsum(returns))
    highs = closes * (1 + np.abs(np.random.normal(0, 0.01, n)))
    lows = closes * (1 - np.abs(np.random.normal(0, 0.01, n)))
    opens = closes * (1 + np.random.normal(0, 0.005, n))
    volumes = np.random.uniform(1000, 50000, n)
    idx = pd.date_range(end=pd.Timestamp.utcnow(), periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": closes,
        "volume": volumes,
    }, index=idx)


# ============================================================
# 1) ADX correctness
# ============================================================

def test_adx_strong_uptrend_di_plus_gt_di_minus():
    """In a steady uptrend, +DI must exceed -DI (old buggy version failed this)."""
    n = 200
    closes = np.linspace(100, 150, n)          # steady rise
    noise = np.random.RandomState(7).uniform(0, 0.2, n)
    close = pd.Series(closes + noise)
    high = close + 0.5
    low = close - 0.5
    adx_df = ta.adx(high, low, close, 14)
    last = adx_df.dropna().iloc[-1]
    assert last["plus_di"] > last["minus_di"], (
        f"+DI ({last['plus_di']:.1f}) must exceed -DI ({last['minus_di']:.1f}) in uptrend"
    )


def test_adx_strong_downtrend_di_minus_gt_di_plus():
    """In a steady downtrend, -DI must exceed +DI."""
    n = 200
    closes = np.linspace(150, 100, n)          # steady fall
    noise = np.random.RandomState(7).uniform(0, 0.2, n)
    close = pd.Series(closes + noise)
    high = close + 0.5
    low = close - 0.5
    adx_df = ta.adx(high, low, close, 14)
    last = adx_df.dropna().iloc[-1]
    assert last["minus_di"] > last["plus_di"], (
        f"-DI ({last['minus_di']:.1f}) must exceed +DI ({last['plus_di']:.1f}) in downtrend"
    )


def test_adx_range_and_strong_trend_values():
    """ADX in a strong one-directional trend should be high (> 25)."""
    n = 200
    closes = np.linspace(100, 160, n)
    close = pd.Series(closes)
    high = close + 0.5
    low = close - 0.5
    adx_df = ta.adx(high, low, close, 14)
    last = adx_df.dropna().iloc[-1]
    assert 0 <= last["adx"] <= 100
    assert last["adx"] > 25, f"ADX should be high in strong trend, got {last['adx']:.1f}"


# ============================================================
# 2) Confidence model
# ============================================================

def _make_signal(score, name="s"):
    from src.strategies.base import Signal
    return Signal(strategy=name, direction="bullish" if score > 0 else "bearish",
                  score=score, confidence=abs(score) / 100)


def test_confidence_neutral_is_zero_not_fifty():
    """Neutral market must give ~0% confidence (old formula gave 50%)."""
    from src.analysis.scorer import SignalScorer
    s = SignalScorer()
    signals = [_make_signal(0), _make_signal(0), _make_signal(0)]
    verdict = s._compute_confidence(signals, s.strategies)
    assert verdict["direction"] == "neutral"
    assert verdict["confidence"] == 0.0


def test_confidence_scales_with_confluence():
    """More agreeing strategies => higher confidence (monotonic)."""
    from src.analysis.scorer import SignalScorer
    s = SignalScorer()
    one = s._compute_confidence([_make_signal(80), _make_signal(0), _make_signal(0)], s.strategies)
    two = s._compute_confidence([_make_signal(80), _make_signal(70), _make_signal(0)], s.strategies)
    three = s._compute_confidence([_make_signal(80), _make_signal(70), _make_signal(65)], s.strategies)
    assert one["confidence"] < two["confidence"] < three["confidence"]
    assert one["confidence"] > 55   # single strong strategy is already meaningful
    assert three["confidence"] < 100


def test_confidence_opposing_signals_reduced():
    """A strong opposing vote must reduce confidence vs unanimous."""
    from src.analysis.scorer import SignalScorer
    s = SignalScorer()
    unanimous = s._compute_confidence([_make_signal(80), _make_signal(70), _make_signal(65)], s.strategies)
    opposed = s._compute_confidence([_make_signal(80), _make_signal(-70), _make_signal(65)], s.strategies)
    assert opposed["confidence"] < unanimous["confidence"]


# ============================================================
# 3) Scorer end-to-end output validity
# ============================================================

def test_scorer_output_structure_valid():
    """For a trending sample the scorer must return a coherent recommendation."""
    from src.analysis.scorer import scorer
    df = _sample_ohlcv(300, trend=0.004)
    rec = scorer.analyze_symbol(df, "TESTUSDT")
    assert rec.get("direction") in ("bullish", "bearish", "neutral")
    if rec["direction"] == "bullish":
        assert rec["stop_loss"] < rec["current_price"] < rec["take_profit"]
        assert rec["risk_reward_ratio"] >= 1.2
    assert 0 <= rec["confidence"] <= 100
    assert rec["expected_rise_pct"] >= 0


def test_scorer_insufficient_data():
    from src.analysis.scorer import scorer
    df = _sample_ohlcv(30)
    rec = scorer.analyze_symbol(df, "TESTUSDT")
    assert rec.get("skip") is True


# ============================================================
# 4) Trailing ladder: direct jump
# ============================================================

def test_trailing_ladder_jumps_directly():
    """At +3.5% profit the SL must jump straight to entry*1.02 (not 1 level/cycle)."""
    from src.risk.manager import RiskManager
    rm = RiskManager(capital=10000)
    rm.open_positions = [{
        "symbol": "TESTUSDT", "direction": "bullish",
        "entry_price": 100.0, "stop_loss": 95.0, "take_profit": 110.0,
        "size": 1.0, "notional_usd": 100.0, "entry_fee": 0.1,
        "entry_time": datetime.now(timezone.utc).isoformat(),
        "status": "open", "paper": True,
    }]
    prices = {"TESTUSDT": 103.5}
    updates = rm.apply_trailing_logic(prices, {})
    assert updates, "expected a trailing update"
    pos = rm.open_positions[0]
    assert pos["stop_loss"] >= 102.0, (
        f"SL should jump to >= entry*1.02 = 102.0, got {pos['stop_loss']}"
    )


def test_trailing_never_loosens_sl():
    from src.risk.manager import RiskManager
    rm = RiskManager(capital=10000)
    rm.open_positions = [{
        "symbol": "TESTUSDT", "direction": "bullish",
        "entry_price": 100.0, "stop_loss": 103.0, "take_profit": 110.0,
        "size": 1.0, "notional_usd": 100.0, "entry_fee": 0.1,
        "entry_time": datetime.now(timezone.utc).isoformat(),
        "status": "open", "paper": True,
    }]
    # price dips back to +1% level; SL must stay at 103
    rm.apply_trailing_logic({"TESTUSDT": 101.0}, {})
    assert rm.open_positions[0]["stop_loss"] == 103.0


# ============================================================
# 5) Duplicate position guard
# ============================================================

def test_open_paper_position_rejects_duplicate_symbol(tmp_path, monkeypatch):
    from src.risk import manager as mgr
    monkeypatch.setattr(mgr, "POSITIONS_FILE", tmp_path / "positions.json")
    rm = mgr.RiskManager(capital=10000)
    rm.open_positions = [{
        "symbol": "BTCUSDT", "direction": "bullish",
        "entry_price": 50000, "stop_loss": 48000, "take_profit": 55000,
        "size": 0.001, "notional_usd": 50, "entry_fee": 0.05,
        "entry_time": datetime.now(timezone.utc).isoformat(),
        "status": "open", "paper": True,
    }]
    rec = {
        "symbol": "BTCUSDT", "direction": "bullish",
        "current_price": 51000, "stop_loss": 49000, "take_profit": 56000,
        "confidence": 80, "expected_rise_pct": 3.0, "risk_reward_ratio": 2.0,
    }
    result = rm.open_paper_position(rec)
    assert result["status"] == "rejected"
    assert any("already has an open position" in r for r in result["reasons"])


# ============================================================
# 6) Telegram cooldown
# ============================================================

def test_telegram_cooldown_blocks_recent_symbol():
    from src.notifications.telegram_bot import TelegramNotifier
    tn = TelegramNotifier()
    tn._sent_at = {"BTCUSDT": now_utc().isoformat()}
    assert tn._in_cooldown("BTCUSDT") is True
    assert tn._in_cooldown("ETHUSDT") is False


def test_telegram_cooldown_expires():
    from src.notifications.telegram_bot import TelegramNotifier
    tn = TelegramNotifier()
    old = now_utc() - timedelta(hours=10)  # > default 4h cooldown
    tn._sent_at = {"BTCUSDT": old.isoformat()}
    assert tn._in_cooldown("BTCUSDT") is False
