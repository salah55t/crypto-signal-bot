"""
Tests for indicators module.
Run: pytest tests/test_indicators.py -v
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.indicators import technical as ta
from src.indicators import volume as vol
from src.indicators import patterns as pat
from src.indicators import liquidity as liq


def _sample_ohlcv(n=200, seed=42):
    """Generate synthetic OHLCV data for tests."""
    np.random.seed(seed)
    base = 100
    returns = np.random.normal(0.0005, 0.02, n)
    closes = base * np.exp(np.cumsum(returns))
    highs = closes * (1 + np.abs(np.random.normal(0, 0.01, n)))
    lows = closes * (1 - np.abs(np.random.normal(0, 0.01, n)))
    opens = closes * (1 + np.random.normal(0, 0.005, n))
    volumes = np.random.uniform(1000, 50000, n)
    idx = pd.date_range(end=pd.Timestamp.utcnow(), periods=n, freq='1h', tz='UTC')
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": closes,
        "volume": volumes
    }, index=idx)


def test_rsi_range():
    df = _sample_ohlcv()
    r = ta.rsi(df["close"], 14)
    assert r.notna().sum() > 0
    assert r.min() >= 0 and r.max() <= 100


def test_macd_columns():
    df = _sample_ohlcv()
    m = ta.macd(df["close"])
    assert set(m.columns) == {"macd", "signal", "histogram"}


def test_bollinger_bands():
    df = _sample_ohlcv()
    bb = ta.bollinger_bands(df["close"], 20, 2)
    assert "upper" in bb.columns and "lower" in bb.columns
    # upper > middle > lower (when std > 0)
    last = bb.dropna().iloc[-1]
    assert last["upper"] >= last["middle"] >= last["lower"]


def test_atr_positive():
    df = _sample_ohlcv()
    a = ta.atr(df["high"], df["low"], df["close"], 14)
    assert (a.dropna() >= 0).all()


def test_volume_analysis():
    df = _sample_ohlcv()
    a = vol.analyze_volume(df, period=20)
    assert "current_volume" in a
    assert "spike_ratio" in a
    assert "cvd" in a


def test_pattern_detection_no_crash():
    df = _sample_ohlcv()
    pats = pat.detect_all_patterns(df)
    assert isinstance(pats, list)


def test_support_resistance():
    df = _sample_ohlcv()
    sr = liq.find_support_resistance(df, lookback=50)
    assert "supports" in sr
    assert "resistances" in sr
