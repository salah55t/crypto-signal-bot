"""Strategies package — Clean & focused.

The bot uses ONLY 3 powerful composite strategies, each combining
multiple indicators like a real trader would:

1. TrendPullbackStrategy (weight 2.0)
   - Buy strength on pullbacks in strong uptrends
   - EMA stack + ADX + pullback to EMA 21 + bullish reversal candle
   - The classic trend-following setup

2. LiquiditySweepReversalStrategy (weight 2.0)
   - Join institutions at stop hunts (ICT methodology)
   - Liquidity Sweep + RSI oversold + reversal candle + volume climax
   - Highest-probability reversal pattern

3. VolatilityBreakoutStrategy (weight 1.8)
   - Trade the squeeze breakout with volume confirmation
   - BB squeeze + volume buildup + breakout candle + ADX rising
   - Backtested 7.5 years on BTC (Reddit r/algotrading)

Legacy strategies (technical, volume, momentum, etc.) are in legacy/
folder for reference but no longer used.

Reinforcement Learning (optional, advanced):
  - Enable with ENABLE_RL_STRATEGY=true
  - Requires stable-baselines3 + trained models
"""
from .base import BaseStrategy, Signal
from .trend_pullback_strategy import TrendPullbackStrategy
from .liquidity_sweep_reversal_strategy import LiquiditySweepReversalStrategy
from .volatility_breakout_strategy import VolatilityBreakoutStrategy

__all__ = [
    "BaseStrategy", "Signal",
    # The 3 composite strategies
    "TrendPullbackStrategy",
    "LiquiditySweepReversalStrategy",
    "VolatilityBreakoutStrategy",
]
