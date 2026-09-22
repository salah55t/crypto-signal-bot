"""
Reinforcement Learning Ensemble Strategy.

Loads DQN, PPO, A2C models trained on crypto data and combines their
predictions via majority voting. This is an ADVANCED strategy that requires
pre-trained models (run scripts/train_rl_agents.py first).

If models are not available, returns neutral signal.

Optional dependency: stable-baselines3, gymnasium, torch
Install: pip install stable-baselines3[extra] gymnasium torch
"""
import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Optional
from config.settings import settings, PROJECT_ROOT
from src.indicators.technical import rsi, macd, bollinger_bands, atr
from src.utils.logger import log
from src.strategies.base import BaseStrategy, Signal

MODEL_PATHS = {
    "DQN": PROJECT_ROOT / "data" / "models" / "dqn_crypto.zip",
    "PPO": PROJECT_ROOT / "data" / "models" / "ppo_crypto.zip",
    "A2C": PROJECT_ROOT / "data" / "models" / "a2c_crypto.zip",
}


class RLStrategy(BaseStrategy):
    """RL Ensemble Strategy (DQN + PPO + A2C majority voting)."""
    name = "rl_ensemble"
    weight = 1.5  # heavy weight - state-of-the-art ML

    def __init__(self, weight: float = None):
        super().__init__(weight)
        self.models = {}
        self.enabled = os.getenv("ENABLE_RL_STRATEGY", "false").lower() == "true"
        self._load_models()

    def _load_models(self):
        """Load all three RL models if available."""
        if not self.enabled:
            log.info("[yellow]RL strategy disabled[/] (set ENABLE_RL_STRATEGY=true to enable)")
            return

        if not self._check_dependencies():
            log.warning("[yellow]RL dependencies missing[/] - install stable-baselines3")
            self.enabled = False
            return

        try:
            from stable_baselines3 import DQN, PPO, A2C
            loaders = {"DQN": DQN, "PPO": PPO, "A2C": A2C}
            loaded = 0
            for name, path in MODEL_PATHS.items():
                if path.exists():
                    try:
                        self.models[name] = loaders[name].load(str(path))
                        log.info(f"[green]{name} model loaded[/] from {path}")
                        loaded += 1
                    except Exception as e:
                        log.warning(f"Failed to load {name} from {path}: {e}")
            if loaded == 0:
                log.warning(
                    "[yellow]No RL models found[/]. Run `python scripts/train_rl_agents.py` to train."
                )
                self.enabled = False
            elif loaded < 3:
                log.warning(
                    f"[yellow]Only {loaded}/3 RL models loaded[/]. Ensemble will be limited."
                )
        except Exception as e:
            log.error(f"Failed to load RL models: {e}")
            self.enabled = False

    def _check_dependencies(self) -> bool:
        """Check if stable-baselines3 is installed."""
        try:
            import stable_baselines3
            import gymnasium
            import torch
            return True
        except ImportError:
            return False

    def _build_observation(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Build the 10-feature observation vector from latest candle."""
        if len(df) < 60:
            return None
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        # Compute features (must match rl_environment.py)
        rsi_val = float(rsi(close, 14).iloc[-1]) / 100.0
        macd_df = macd(close, 12, 26, 9)
        atr_val = float(atr(high, low, close, 14).iloc[-1])
        macd_hist = float(macd_df["histogram"].iloc[-1])
        macd_norm = macd_hist / atr_val if atr_val > 0 else 0

        bb = bollinger_bands(close, 20, 2)
        bb_pct = float(bb["percent_b"].iloc[-1])
        bb_width = float(bb["bandwidth"].iloc[-1] * 100)

        vol_sma = volume.rolling(20).mean().iloc[-1]
        vol_ratio = (float(volume.iloc[-1]) / vol_sma if vol_sma > 0 else 1.0) / 5.0

        atr_pct = float(atr_val / close.iloc[-1]) if close.iloc[-1] > 0 else 0

        daily_return = float(close.pct_change().iloc[-1])
        # Market regime using EMA cross
        ema_short = close.ewm(span=20).mean().iloc[-1]
        ema_long = close.ewm(span=50).mean().iloc[-1]
        regime = 1.0 if ema_short > ema_long else (-1.0 if ema_short < ema_long else 0.0)

        # Position state and unrealized PnL are unknown for fresh observation
        # so we set to flat (0) and 0 unrealized PnL
        obs = np.array([
            daily_return,
            rsi_val - 0.5,
            macd_norm,
            bb_pct - 0.5,
            bb_width,
            vol_ratio - 0.5,
            atr_pct,
            regime,
            0.0,  # position state (unknown)
            0.0,  # unrealized PnL (unknown)
        ], dtype=np.float32)
        return np.clip(obs, -10, 10)

    def analyze(self, df: pd.DataFrame, symbol: str,
                multi_tf_data: Optional[Dict[str, pd.DataFrame]] = None,
                order_book: Optional[Dict] = None) -> Signal:
        if not self.enabled or not self.models:
            return self._neutral("RL strategy disabled or no models loaded")

        if len(df) < 60:
            return self._neutral("Insufficient data")

        try:
            obs = self._build_observation(df)
            if obs is None:
                return self._neutral("Failed to build observation")

            # Get predictions from each model
            votes = {}
            probs = {}
            for name, model in self.models.items():
                try:
                    action, _ = model.predict(obs, deterministic=True)
                    votes[name] = int(action)
                    probs[name] = int(action)
                except Exception as e:
                    log.debug(f"{name} prediction failed: {e}")

            if not votes:
                return self._neutral("No predictions from RL models")

            # Majority vote: BUY if >= 2 vote BUY
            buy_votes = sum(1 for v in votes.values() if v == 1)
            sell_votes = sum(1 for v in votes.values() if v == 2)
            hold_votes = sum(1 for v in votes.values() if v == 0)

            details = {
                "votes": votes,
                "buy_votes": buy_votes,
                "sell_votes": sell_votes,
                "hold_votes": hold_votes,
                "total_models": len(votes),
            }

            # Strong buy signal: 2+ models agree on BUY
            if buy_votes >= 2:
                # Stronger if all 3 agree
                strength = 80 if buy_votes == 3 else 60
                reasons = [
                    f"{buy_votes}/{len(votes)} RL models voted BUY",
                    *[f"{name}: action={action}" for name, action in votes.items()]
                ]
                return Signal(
                    strategy=self.name,
                    direction="bullish",
                    score=strength,
                    confidence=strength / 100.0,
                    reasons=reasons,
                    details=details,
                )
            # Strong sell signal: 2+ models agree on SELL
            elif sell_votes >= 2:
                strength = 80 if sell_votes == 3 else 60
                reasons = [
                    f"{sell_votes}/{len(votes)} RL models voted SELL",
                    *[f"{name}: action={action}" for name, action in votes.items()]
                ]
                return Signal(
                    strategy=self.name,
                    direction="bearish",
                    score=-strength,
                    confidence=strength / 100.0,
                    reasons=reasons,
                    details=details,
                )
            # Mixed or HOLD
            return self._neutral(
                f"RL vote mixed: BUY={buy_votes}, SELL={sell_votes}, HOLD={hold_votes}",
                details
            )

        except Exception as e:
            log.error(f"RL analyze error: {e}")
            return self._neutral(f"RL error: {e}")
