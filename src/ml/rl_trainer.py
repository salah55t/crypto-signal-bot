"""
Reinforcement Learning Trainer for Crypto Trading Agents.

Trains three RL algorithms (DQN, PPO, A2C) using stable-baselines3
on the custom CryptoTradingEnv. Each agent learns its own trading policy,
then they are combined via ensemble voting for robust predictions.

Usage:
  python scripts/train_rl_agents.py --symbol BTCUSDT --interval 1h --limit 2000

Output: 3 model files in data/models/:
  - dqn_crypto.zip
  - ppo_crypto.zip
  - a2c_crypto.zip

Note: stable-baselines3 is a heavy dependency (~500MB with PyTorch).
Install with: pip install stable-baselines3[extra] gymnasium
"""
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

from config.settings import settings, PROJECT_ROOT
from src.core.data_fetcher import data_fetcher
from src.ml.rl_environment import CryptoTradingEnv
from src.utils.logger import log
from src.utils.helpers import save_json, to_json_safe, now_utc

MODELS_DIR = PROJECT_ROOT / "data" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def check_rl_dependencies() -> bool:
    """Check if stable-baselines3 is installed."""
    try:
        import stable_baselines3
        import gymnasium
        import torch
        return True
    except ImportError as e:
        log.error(
            f"[red]RL dependencies not installed[/]: {e}\n"
            f"Install with: pip install stable-baselines3[extra] gymnasium torch"
        )
        return False


def train_agents(df: pd.DataFrame, symbol: str = "BTCUSDT",
                  total_timesteps: int = 50000,
                  algorithms: List[str] = None) -> Dict:
    """
    Train multiple RL agents on the given data.
    Returns dict with paths to saved models and training metrics.
    """
    if not check_rl_dependencies():
        return {"status": "failed", "reason": "dependencies_missing"}

    algorithms = algorithms or ["DQN", "PPO", "A2C"]
    from stable_baselines3 import DQN, PPO, A2C
    from stable_baselines3.common.callbacks import BaseCallback

    log.info(f"[cyan]Training RL agents[/] on {symbol} ({len(df)} bars)")
    log.info(f"Algorithms: {algorithms}, timesteps: {total_timesteps}")

    # Callback to track training progress
    class TrainingLogger(BaseCallback):
        def __init__(self, log_interval: int = 1000):
            super().__init__()
            self.log_interval = log_interval
            self.episode_rewards = []

        def _on_step(self) -> bool:
            if self.n_calls % self.log_interval == 0:
                log.info(f"Training step {self.n_calls}/{total_timesteps}")
            return True

    env = CryptoTradingEnv(df, initial_capital=10000, transaction_cost=0.001)
    log.info(f"Environment created: {len(df)} steps, action_space={env.action_space}")

    results = {}

    # Train DQN
    if "DQN" in algorithms:
        log.info("\n[bold cyan]Training DQN...[/]")
        try:
            model = DQN(
                "MlpPolicy", env,
                learning_rate=0.0003,
                buffer_size=10000,
                batch_size=64,
                learning_starts=500,
                verbose=0,
                seed=42
            )
            callback = TrainingLogger(log_interval=2000)
            model.learn(total_timesteps=total_timesteps, callback=callback)
            save_path = MODELS_DIR / "dqn_crypto.zip"
            model.save(str(save_path))
            log.info(f"[green]DQN saved[/] to {save_path}")
            results["DQN"] = {"status": "success", "path": str(save_path)}
        except Exception as e:
            log.error(f"DQN training failed: {e}")
            results["DQN"] = {"status": "failed", "error": str(e)}

    # Train PPO
    if "PPO" in algorithms:
        log.info("\n[bold cyan]Training PPO...[/]")
        try:
            model = PPO(
                "MlpPolicy", env,
                learning_rate=0.0003,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                verbose=0,
                seed=42
            )
            callback = TrainingLogger(log_interval=2000)
            model.learn(total_timesteps=total_timesteps, callback=callback)
            save_path = MODELS_DIR / "ppo_crypto.zip"
            model.save(str(save_path))
            log.info(f"[green]PPO saved[/] to {save_path}")
            results["PPO"] = {"status": "success", "path": str(save_path)}
        except Exception as e:
            log.error(f"PPO training failed: {e}")
            results["PPO"] = {"status": "failed", "error": str(e)}

    # Train A2C
    if "A2C" in algorithms:
        log.info("\n[bold cyan]Training A2C...[/]")
        try:
            model = A2C(
                "MlpPolicy", env,
                learning_rate=0.0007,
                n_steps=5,
                gamma=0.99,
                verbose=0,
                seed=42
            )
            callback = TrainingLogger(log_interval=2000)
            model.learn(total_timesteps=total_timesteps, callback=callback)
            save_path = MODELS_DIR / "a2c_crypto.zip"
            model.save(str(save_path))
            log.info(f"[green]A2C saved[/] to {save_path}")
            results["A2C"] = {"status": "success", "path": str(save_path)}
        except Exception as e:
            log.error(f"A2C training failed: {e}")
            results["A2C"] = {"status": "failed", "error": str(e)}

    # Evaluate each model on the training data (backtest)
    log.info("\n[bold cyan]Evaluating agents on training data...[/]")
    eval_results = {}
    for algo_name, info in results.items():
        if info.get("status") != "success":
            continue
        try:
            eval_env = CryptoTradingEnv(df)
            obs, _ = eval_env.reset()
            done = False
            while not done:
                # Load model
                if algo_name == "DQN":
                    model = DQN.load(info["path"])
                elif algo_name == "PPO":
                    model = PPO.load(info["path"])
                elif algo_name == "A2C":
                    model = A2C.load(info["path"])
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info_eval = eval_env.step(action)
                done = done or truncated
            log.info(
                f"{algo_name} final capital: ${info_eval.get('final_capital', 0):.2f} "
                f"(return: {info_eval.get('total_return_pct', 0):+.2f}%)"
            )
            eval_results[algo_name] = {
                "final_capital": float(info_eval.get("final_capital", 0)),
                "total_return_pct": float(info_eval.get("total_return_pct", 0)),
                "max_drawdown_pct": float(info_eval.get("max_drawdown_pct", 0)),
                "total_trades": int(info_eval.get("total_trades", 0)),
                "win_rate": float(info_eval.get("win_rate", 0)),
            }
        except Exception as e:
            log.error(f"Evaluation failed for {algo_name}: {e}")

    # Save training report
    report = {
        "timestamp": now_utc().isoformat(),
        "symbol": symbol,
        "bars": len(df),
        "total_timesteps": total_timesteps,
        "results": results,
        "evaluations": eval_results,
    }
    report_path = MODELS_DIR / "rl_training_report.json"
    save_json(to_json_safe(report), report_path)
    log.info(f"Training report saved to {report_path}")

    return report


def ensemble_predict(obs: np.ndarray, models: Dict) -> Tuple[int, Dict]:
    """
    Run ensemble prediction across multiple RL models.
    Returns (action, probabilities_dict).

    Voting rule: majority vote (>=2 of 3 agree).
    If all disagree, return HOLD (0).
    """
    votes = {}
    probabilities = {}
    for name, model in models.items():
        try:
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            votes[name] = action
            probabilities[name] = action
        except Exception as e:
            log.debug(f"{name} predict failed: {e}")

    if not votes:
        return 0, {"error": "no_predictions"}

    # Majority vote
    from collections import Counter
    counts = Counter(votes.values())
    most_common_action, most_common_count = counts.most_common(1)[0]

    if most_common_count >= max(2, len(votes) // 2 + 1):
        return most_common_action, probabilities
    # No majority - return HOLD
    return 0, probabilities
