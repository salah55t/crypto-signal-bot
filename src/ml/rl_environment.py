"""
Custom Gymnasium Trading Environment for Cryptocurrency Trading.

This environment simulates crypto trading with:
  - Discrete action space: 0=HOLD, 1=BUY, 2=SELL
  - Observation: 10 normalized technical features + portfolio state
  - Reward: PnL - transaction costs - drawdown penalty

Inspired by the Multi-Agent RL Trading System architecture,
adapted for crypto markets with 24/7 trading and higher volatility.
"""
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Dict, Any
import gymnasium as gym
from gymnasium import spaces

from src.indicators.technical import rsi, macd, bollinger_bands, atr
from src.utils.logger import log


class CryptoTradingEnv(gym.Env):
    """
    Custom trading environment for crypto.

    Action space: Discrete(3)
        0 = HOLD (no action)
        1 = BUY  (open/keep long position)
        2 = SELL (close position / stay flat)

    Observation space: Box(10,)
        0: Daily return (normalized)
        1: RSI 14 / 100
        2: MACD histogram (normalized by ATR)
        3: Bollinger %B
        4: Bollinger bandwidth (normalized)
        5: Volume ratio (current / SMA)
        6: ATR % (volatility)
        7: Market regime (1=uptrend, -1=downtrend, 0=range)
        8: Position state (1=long, 0=flat)
        9: Unrealized PnL % (normalized)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, df: pd.DataFrame, initial_capital: float = 10000.0,
                 transaction_cost: float = 0.001,  # 0.1% per trade
                 max_position_pct: float = 0.95,
                 reward_scale: float = 100.0,
                 render_mode: Optional[str] = None):
        super().__init__()
        if len(df) < 100:
            raise ValueError(f"Need at least 100 candles, got {len(df)}")

        self.df = df.copy().reset_index(drop=True)
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.max_position_pct = max_position_pct
        self.reward_scale = reward_scale
        self.render_mode = render_mode

        # Action space: 0=HOLD, 1=BUY, 2=SELL
        self.action_space = spaces.Discrete(3)

        # Observation space: 10 features, all normalized to [-1, 1] roughly
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
        )

        # Pre-compute indicators
        self._prepare_data()

        # Episode state
        self.current_step = 0
        self.position = 0  # 0=flat, 1=long
        self.entry_price = 0.0
        self.capital = initial_capital
        self.position_size = 0.0
        self.peak_capital = initial_capital
        self.total_trades = 0
        self.winning_trades = 0
        self.peak_drawdown = 0.0

    def _prepare_data(self):
        """Pre-compute all indicators needed for observation."""
        df = self.df
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # RSI
        df["rsi"] = rsi(close, 14) / 100.0

        # MACD
        macd_df = macd(close, 12, 26, 9)
        atr_val = atr(high, low, close, 14)
        # Normalize MACD by ATR (so it's scale-invariant)
        df["macd_norm"] = (macd_df["histogram"] / atr_val.replace(0, np.nan)).fillna(0)

        # Bollinger
        bb = bollinger_bands(close, 20, 2)
        df["bb_pct"] = bb["percent_b"].fillna(0.5).clip(0, 1)
        # Bandwidth normalized (clip at 0.1 to avoid outliers)
        df["bb_width"] = (bb["bandwidth"] * 100).clip(0, 0.1)

        # Volume ratio
        vol_sma = volume.rolling(20).mean()
        df["vol_ratio"] = (volume / vol_sma.replace(0, np.nan)).fillna(1.0).clip(0, 5) / 5.0

        # ATR as % of price
        df["atr_pct"] = (atr_val / close).fillna(0).clip(0, 0.1)

        # Daily return
        df["return"] = close.pct_change().fillna(0).clip(-0.2, 0.2)

        # Market regime (using EMA cross)
        ema_short = close.ewm(span=20).mean()
        ema_long = close.ewm(span=50).mean()
        df["regime"] = np.where(ema_short > ema_long, 1.0,
                                np.where(ema_short < ema_long, -1.0, 0.0))

        # Drop NaN rows
        self.df = df.dropna().reset_index(drop=True)

    def _get_observation(self) -> np.ndarray:
        """Build the observation vector at current step."""
        if self.current_step >= len(self.df):
            return np.zeros(10, dtype=np.float32)

        row = self.df.iloc[self.current_step]
        unrealized_pnl = 0.0
        if self.position == 1 and self.entry_price > 0:
            unrealized_pnl = (row["close"] - self.entry_price) / self.entry_price

        obs = np.array([
            row["return"],
            row["rsi"] - 0.5,  # center around 0
            row["macd_norm"],
            row["bb_pct"] - 0.5,  # center around 0
            row["bb_width"],
            row["vol_ratio"] - 0.5,
            row["atr_pct"],
            row["regime"],
            float(self.position),
            unrealized_pnl,
        ], dtype=np.float32)

        # Clip to reasonable bounds
        return np.clip(obs, -10, 10)

    def _take_action(self, action: int) -> Tuple[float, Dict]:
        """Execute an action and return reward + info."""
        if self.current_step >= len(self.df) - 1:
            return 0.0, {"done": True}

        current_price = float(self.df.iloc[self.current_step]["close"])
        next_price = float(self.df.iloc[self.current_step + 1]["close"])

        reward = 0.0
        info = {
            "action": action,
            "current_price": current_price,
            "position_before": self.position,
        }

        # Action: 0=HOLD, 1=BUY, 2=SELL
        if action == 1:  # BUY
            if self.position == 0:  # open new long
                self.position = 1
                self.entry_price = current_price * (1 + self.transaction_cost)
                self.position_size = (self.capital * self.max_position_pct) / current_price
                self.total_trades += 1
            # If already long, do nothing (no averaging in)
        elif action == 2:  # SELL
            if self.position == 1:  # close long
                exit_price = current_price * (1 - self.transaction_cost)
                pnl = (exit_price - self.entry_price) * self.position_size
                self.capital += pnl
                if pnl > 0:
                    self.winning_trades += 1
                self.position = 0
                self.entry_price = 0.0
                self.position_size = 0.0
                self.total_trades += 1
                info["pnl"] = float(pnl)
                reward = pnl / self.initial_capital * self.reward_scale
        # action == 0 (HOLD): do nothing

        # Mark-to-market reward (small reward/penalty based on position performance)
        if self.position == 1:
            mtm_pnl = (next_price - self.entry_price) / self.entry_price
            # Small incremental reward for holding a winning position
            reward += mtm_pnl * self.reward_scale * 0.1

        # Update peak capital & drawdown
        current_equity = self.capital
        if self.position == 1:
            current_equity += (next_price - self.entry_price) * self.position_size
        if current_equity > self.peak_capital:
            self.peak_capital = current_equity
        drawdown = (self.peak_capital - current_equity) / self.peak_capital
        if drawdown > self.peak_drawdown:
            self.peak_drawdown = drawdown

        # Drawdown penalty
        if drawdown > 0.05:  # 5%+ drawdown
            reward -= (drawdown - 0.05) * self.reward_scale * 2

        info["equity"] = float(current_equity)
        info["drawdown"] = float(drawdown)
        info["position_after"] = self.position
        return float(reward), info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take one step in the environment."""
        reward, info = self._take_action(int(action))
        self.current_step += 1

        terminated = self.current_step >= len(self.df) - 1
        truncated = False

        # At episode end, close any open position
        if terminated and self.position == 1:
            last_price = float(self.df.iloc[-1]["close"])
            exit_price = last_price * (1 - self.transaction_cost)
            pnl = (exit_price - self.entry_price) * self.position_size
            self.capital += pnl
            if pnl > 0:
                self.winning_trades += 1
            info["final_pnl"] = float(pnl)
            self.position = 0
            self.total_trades += 1

        obs = self._get_observation()
        info["total_trades"] = self.total_trades
        info["win_rate"] = self.winning_trades / max(1, self.total_trades)
        info["final_capital"] = self.capital
        info["total_return_pct"] = (self.capital - self.initial_capital) / self.initial_capital * 100
        info["max_drawdown_pct"] = self.peak_drawdown * 100

        return obs, reward, terminated, truncated, info

    def reset(self, *, seed: Optional[int] = None,
              options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        self.current_step = 0
        self.position = 0
        self.entry_price = 0.0
        self.capital = self.initial_capital
        self.position_size = 0.0
        self.peak_capital = self.initial_capital
        self.total_trades = 0
        self.winning_trades = 0
        self.peak_drawdown = 0.0
        return self._get_observation(), {}

    def render(self):
        """Render info to console."""
        if self.render_mode == "human" and self.current_step % 50 == 0:
            log.info(
                f"Step {self.current_step}/{len(self.df)} | "
                f"Capital: ${self.capital:.2f} | "
                f"Position: {'LONG' if self.position else 'FLAT'} | "
                f"Drawdown: {self.peak_drawdown*100:.2f}%"
            )
