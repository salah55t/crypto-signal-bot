"""Strategies package — v5.7 Signal Stack (6 composite strategies).

The bot uses SIX composite strategies. Every strategy follows the
"Signal Stack Framework" golden rule: combine ONE indicator from each
class - Direction / Momentum / Volume-Volatility - never stack same-class
indicators (RSI+Stoch+MACD together = one redundant vote).

Original trio:
1. TrendPullbackStrategy (weight 2.0)
   - Buy strength on pullbacks in strong uptrends
   - EMA stack + ADX + pullback to EMA 21 + bullish reversal candle

2. LiquiditySweepReversalStrategy (weight 2.0)
   - Join institutions at stop hunts (ICT methodology)
   - Liquidity Sweep + RSI oversold + reversal candle + volume climax

3. VolatilityBreakoutStrategy (weight 1.8)
   - Trade the squeeze breakout with volume confirmation
   - BB squeeze + volume buildup + breakout candle + ADX rising

v5.7 user-specified trio (Signal Stack):
4. TripleConfluenceTrendStrategy (weight 1.6)
   - Ride extended trend waves: EMA200/EMA50 (dir) + RSI 14 (mom)
     + Volume/OBV (liq). Exit: EMA50 break, or RSI>70 + bearish divergence.

5. BBMeanReversionStrategy (weight 1.2)
   - Fade band extremes back to the mean: BB(20,2) (vol) + Stochastic
     (14,3,3) (mom) + bullish candle/volume (confirm).
     TP1 = middle band concept, full exit at upper band + RSI>70.

6. MACDBreakoutStrategy (weight 1.4)
   - Ride momentum bursts: EMA50 (dir) + MACD(12,26,9) (mom) + volume (liq).
     Zero-line cross + histogram flip. Laddered multi-TP + trailing SL are
     the bot's own TP1/TP2 + chandelier machinery; exit when MACD bends down.

Legacy strategies (technical, volume, momentum, etc.) are in legacy/
folder for reference but no longer used.
"""
from .base import BaseStrategy, Signal
from .trend_pullback_strategy import TrendPullbackStrategy
from .liquidity_sweep_reversal_strategy import LiquiditySweepReversalStrategy
from .volatility_breakout_strategy import VolatilityBreakoutStrategy
from .triple_confluence_trend_strategy import TripleConfluenceTrendStrategy
from .bb_mean_reversion_strategy import BBMeanReversionStrategy
from .macd_breakout_strategy import MACDBreakoutStrategy

__all__ = [
    "BaseStrategy", "Signal",
    # Original trio
    "TrendPullbackStrategy",
    "LiquiditySweepReversalStrategy",
    "VolatilityBreakoutStrategy",
    # v5.7 Signal Stack trio
    "TripleConfluenceTrendStrategy",
    "BBMeanReversionStrategy",
    "MACDBreakoutStrategy",
]
