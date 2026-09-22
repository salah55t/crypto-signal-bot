"""
Crypto Signal Bot - RL Trainer for Hugging Face Spaces

A Gradio web app that allows training DQN, PPO, A2C RL agents on
crypto data from Binance's public data endpoint (data-api.binance.vision)
which is NOT geo-restricted.

Run locally:
  pip install -r requirements.txt
  python app.py

Deploy on Hugging Face Spaces (free CPU tier or ZeroGPU):
  1. Create new Space at https://huggingface.co/new-space
  2. Choose SDK: Gradio
  3. Upload all files from hf-spaces-trainer/
  4. Spaces will auto-install from requirements.txt
  5. Visit the Space URL to use the trainer

Author: Crypto Signal Bot
"""
import os
import sys
import json
import time
import tempfile
from pathlib import Path
from datetime import datetime

import gradio as gr
import pandas as pd
import numpy as np
import requests

# ============================================================
# Binance public data fetcher (works from any region including US)
# ============================================================
BINANCE_DATA_URL = "https://data-api.binance.vision"


def fetch_candles(symbol: str, interval: str, limit: int = 2000) -> pd.DataFrame:
    """Fetch historical candles from Binance public data endpoint (no API key, no geo-restriction)."""
    url = f"{BINANCE_DATA_URL}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    raw = r.json()
    if not raw:
        return pd.DataFrame()
    cols = ["open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"]
    df = pd.DataFrame(raw, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    numeric_cols = ["open", "high", "low", "close", "volume",
                    "quote_volume", "trades", "taker_buy_base", "taker_buy_quote"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df.set_index("open_time", inplace=True)
    df.drop(columns=["ignore"], inplace=True)
    return df


def get_top_usdt_pairs(limit: int = 30) -> list:
    """Get top USDT pairs by 24h volume."""
    try:
        r = requests.get(f"{BINANCE_DATA_URL}/api/v3/ticker/24hr", timeout=30)
        r.raise_for_status()
        tickers = r.json()
        pairs = [
            t for t in tickers
            if t.get("symbol", "").endswith("USDT")
            and float(t.get("quoteVolume", 0)) > 0
        ]
        pairs.sort(key=lambda t: float(t.get("quoteVolume", 0)), reverse=True)
        return [t["symbol"] for t in pairs[:limit]]
    except Exception as e:
        print(f"Error fetching pairs: {e}")
        return ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]


# ============================================================
# Technical indicators (lightweight versions for HF Spaces)
# ============================================================
def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def bollinger_bands(close: pd.Series, period: int = 20, std_dev: float = 2.0):
    sma = close.rolling(window=period).mean()
    std = close.rolling(window=period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    return sma, upper, lower


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False, min_periods=period).mean()


# ============================================================
# Custom Trading Environment (simplified for HF Spaces)
# ============================================================
import gymnasium as gym
from gymnasium import spaces


class CryptoTradingEnv(gym.Env):
    """Simplified crypto trading environment for HF Spaces."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, df: pd.DataFrame, initial_capital: float = 10000.0,
                 transaction_cost: float = 0.001, render_mode=None):
        super().__init__()
        if len(df) < 100:
            raise ValueError(f"Need at least 100 candles, got {len(df)}")

        self.df = df.copy().reset_index(drop=True)
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(3)  # 0=HOLD, 1=BUY, 2=SELL
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)

        self._prepare_data()
        self.current_step = 0
        self.position = 0
        self.entry_price = 0.0
        self.capital = initial_capital
        self.position_size = 0.0
        self.peak_capital = initial_capital
        self.total_trades = 0
        self.winning_trades = 0
        self.peak_drawdown = 0.0

    def _prepare_data(self):
        df = self.df
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]

        df["rsi"] = rsi(close, 14) / 100.0
        macd_line, signal_line, macd_hist = macd(close, 12, 26, 9)
        atr_val = atr(high, low, close, 14)
        df["macd_norm"] = (macd_hist / atr_val.replace(0, np.nan)).fillna(0)

        sma, upper, lower = bollinger_bands(close, 20, 2)
        bb_width = (upper - lower) / sma
        df["bb_pct"] = ((close - lower) / (upper - lower).replace(0, np.nan)).fillna(0.5).clip(0, 1)
        df["bb_width"] = (bb_width * 100).clip(0, 0.1)

        vol_sma = volume.rolling(20).mean()
        df["vol_ratio"] = (volume / vol_sma.replace(0, np.nan)).fillna(1.0).clip(0, 5) / 5.0
        df["atr_pct"] = (atr_val / close).fillna(0).clip(0, 0.1)
        df["return"] = close.pct_change().fillna(0).clip(-0.2, 0.2)

        ema_short = close.ewm(span=20).mean()
        ema_long = close.ewm(span=50).mean()
        df["regime"] = np.where(ema_short > ema_long, 1.0,
                                np.where(ema_short < ema_long, -1.0, 0.0))

        self.df = df.dropna().reset_index(drop=True)

    def _get_observation(self) -> np.ndarray:
        if self.current_step >= len(self.df):
            return np.zeros(10, dtype=np.float32)
        row = self.df.iloc[self.current_step]
        unrealized_pnl = 0.0
        if self.position == 1 and self.entry_price > 0:
            unrealized_pnl = (row["close"] - self.entry_price) / self.entry_price
        obs = np.array([
            row["return"],
            row["rsi"] - 0.5,
            row["macd_norm"],
            row["bb_pct"] - 0.5,
            row["bb_width"],
            row["vol_ratio"] - 0.5,
            row["atr_pct"],
            row["regime"],
            float(self.position),
            unrealized_pnl,
        ], dtype=np.float32)
        return np.clip(obs, -10, 10)

    def _take_action(self, action: int):
        if self.current_step >= len(self.df) - 1:
            return 0.0, {}
        current_price = float(self.df.iloc[self.current_step]["close"])
        next_price = float(self.df.iloc[self.current_step + 1]["close"])
        reward = 0.0
        info = {"action": action, "current_price": current_price}

        if action == 1:  # BUY
            if self.position == 0:
                self.position = 1
                self.entry_price = current_price * (1 + self.transaction_cost)
                self.position_size = (self.capital * 0.95) / current_price
                self.total_trades += 1
        elif action == 2:  # SELL
            if self.position == 1:
                exit_price = current_price * (1 - self.transaction_cost)
                pnl = (exit_price - self.entry_price) * self.position_size
                self.capital += pnl
                if pnl > 0:
                    self.winning_trades += 1
                self.position = 0
                self.entry_price = 0.0
                self.position_size = 0.0
                self.total_trades += 1
                reward = pnl / self.initial_capital * 100

        if self.position == 1:
            mtm_pnl = (next_price - self.entry_price) / self.entry_price
            reward += mtm_pnl * 10

        current_equity = self.capital
        if self.position == 1:
            current_equity += (next_price - self.entry_price) * self.position_size
        if current_equity > self.peak_capital:
            self.peak_capital = current_equity
        drawdown = (self.peak_capital - current_equity) / self.peak_capital
        if drawdown > self.peak_drawdown:
            self.peak_drawdown = drawdown
        if drawdown > 0.05:
            reward -= (drawdown - 0.05) * 200

        info["equity"] = float(current_equity)
        info["drawdown"] = float(drawdown)
        return float(reward), info

    def step(self, action):
        reward, info = self._take_action(int(action))
        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated = False
        if terminated and self.position == 1:
            last_price = float(self.df.iloc[-1]["close"])
            exit_price = last_price * (1 - self.transaction_cost)
            pnl = (exit_price - self.entry_price) * self.position_size
            self.capital += pnl
            if pnl > 0:
                self.winning_trades += 1
            self.position = 0
            self.total_trades += 1
        obs = self._get_observation()
        info["total_trades"] = self.total_trades
        info["win_rate"] = self.winning_trades / max(1, self.total_trades)
        info["final_capital"] = self.capital
        info["total_return_pct"] = (self.capital - self.initial_capital) / self.initial_capital * 100
        info["max_drawdown_pct"] = self.peak_drawdown * 100
        return obs, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
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


# ============================================================
# Training function
# ============================================================
def train_rl_agent(symbol: str, interval: str, limit: int,
                   algorithm: str, timesteps: int,
                   learning_rate: float, progress=gr.Progress()):
    """Train a single RL agent and return results + download path."""
    progress(0, desc="Fetching data...")
    try:
        df = fetch_candles(symbol, interval, limit)
    except Exception as e:
        return f"❌ Failed to fetch data: {e}", None, None

    if len(df) < 100:
        return f"❌ Insufficient data: got {len(df)} bars (need 100+)", None, None

    progress(0.1, desc="Creating environment...")
    env = CryptoTradingEnv(df, initial_capital=10000)
    progress(0.15, desc=f"Loading {algorithm}...")

    from stable_baselines3 import DQN, PPO, A2C

    algo_map = {
        "DQN": (DQN, {"learning_rate": learning_rate, "buffer_size": 10000,
                      "batch_size": 64, "learning_starts": 500, "verbose": 0, "seed": 42}),
        "PPO": (PPO, {"learning_rate": learning_rate, "n_steps": 2048,
                      "batch_size": 64, "n_epochs": 10, "gamma": 0.99, "verbose": 0, "seed": 42}),
        "A2C": (A2C, {"learning_rate": learning_rate, "n_steps": 5,
                      "gamma": 0.99, "verbose": 0, "seed": 42}),
    }
    if algorithm not in algo_map:
        return f"❌ Unknown algorithm: {algorithm}", None, None

    algo_class, params = algo_map[algorithm]

    try:
        model = algo_class("MlpPolicy", env, **params)
        progress(0.2, desc=f"Training {algorithm} ({timesteps} steps)...")
        # Simulate progress during training (SB3 doesn't have a progress callback)
        # We'll just call learn() and update progress manually
        chunk_size = max(1000, timesteps // 20)
        completed = 0
        while completed < timesteps:
            chunk = min(chunk_size, timesteps - completed)
            model.learn(total_timesteps=chunk, reset_num_timesteps=False)
            completed += chunk
            progress(0.2 + 0.6 * (completed / timesteps),
                     desc=f"Training {algorithm}: {completed}/{timesteps}")
    except Exception as e:
        return f"❌ Training failed: {e}", None, None

    progress(0.85, desc="Evaluating...")
    # Evaluate
    eval_env = CryptoTradingEnv(df)
    obs, _ = eval_env.reset()
    done = False
    eval_info = {}
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, eval_info = eval_env.step(action)
        done = done or truncated

    progress(0.95, desc="Saving model...")
    # Save to temp file (works on HF Spaces persistent storage too)
    save_dir = Path("/data") if Path("/data").exists() else Path(".")
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / f"{algorithm.lower()}_crypto.zip"
    model.save(str(save_path))

    progress(1.0, desc="Done!")

    report = f"""
## ✅ Training Complete!

### 📊 Training Parameters
- **Symbol**: `{symbol}`
- **Interval**: `{interval}`
- **Bars**: `{len(df)}`
- **Algorithm**: `{algorithm}`
- **Timesteps**: `{timesteps:,}`
- **Learning rate**: `{learning_rate}`

### 📈 Evaluation on Training Data
- **Initial capital**: $10,000.00
- **Final capital**: `${eval_info.get('final_capital', 0):.2f}`
- **Total return**: `{eval_info.get('total_return_pct', 0):+.2f}%`
- **Total trades**: `{eval_info.get('total_trades', 0)}`
- **Win rate**: `{eval_info.get('win_rate', 0)*100:.1f}%`
- **Max drawdown**: `{eval_info.get('max_drawdown_pct', 0):.2f}%`

### 💾 Model saved
File: `{save_path}`

### 📥 Next Steps
1. Download the model file above
2. Place it in your local `data/models/{algorithm.lower()}_crypto.zip`
3. Repeat for the other algorithms (DQN, PPO, A2C)
4. Set `ENABLE_RL_STRATEGY=true` in your `.env`
5. Restart your Crypto Signal Bot
"""
    return report, str(save_path), eval_info


# ============================================================
# Gradio UI
# ============================================================
def create_ui():
    with gr.Blocks(title="Crypto RL Trainer", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
# 🧠 Crypto RL Trainer — Train RL agents for crypto trading

This Hugging Face Space allows you to train Reinforcement Learning agents
(DQN, PPO, A2C) on cryptocurrency data from Binance.

**Data source**: `data-api.binance.vision` (Binance public data endpoint — works from any region including US)

**How to use**:
1. Select a symbol (e.g. BTCUSDT)
2. Choose timeframe (15m, 1h, 4h, 1d)
3. Set number of bars (default 2000 = ~3 months on 1h)
4. Choose algorithm (DQN, PPO, A2C)
5. Set training timesteps (50K = ~5 min, 100K = ~15 min)
6. Click **Start Training**
7. Download the trained model file
8. Repeat for other algorithms

⚠️ **Note**: Training on CPU Free Tier is slow. For production, use ZeroGPU or local GPU.
""")

        with gr.Row():
            with gr.Column():
                symbol = gr.Dropdown(
                    choices=get_top_usdt_pairs(30),
                    value="BTCUSDT",
                    label="Symbol",
                    info="Crypto pair to train on"
                )
                interval = gr.Dropdown(
                    choices=["15m", "30m", "1h", "4h", "1d"],
                    value="1h",
                    label="Timeframe",
                    info="Candle interval"
                )
                limit = gr.Slider(
                    minimum=500, maximum=5000, value=2000, step=500,
                    label="Number of bars",
                    info="More bars = more training data but slower fetch"
                )
                algorithm = gr.Radio(
                    choices=["DQN", "PPO", "A2C"],
                    value="PPO",
                    label="Algorithm",
                    info="PPO = stable, DQN = good for discrete, A2C = fast"
                )
                timesteps = gr.Slider(
                    minimum=10000, maximum=200000, value=50000, step=10000,
                    label="Training timesteps",
                    info="50K = ~5 min on CPU, 100K = ~15 min"
                )
                learning_rate = gr.Slider(
                    minimum=0.0001, maximum=0.001, value=0.0003, step=0.0001,
                    label="Learning rate",
                    info="Lower = more stable but slower"
                )
                train_btn = gr.Button("🚀 Start Training", variant="primary")

            with gr.Column():
                report = gr.Markdown("Ready to train. Configure settings on the left, then click **Start Training**.")
                download = gr.File(label="Download trained model", interactive=False)

        train_btn.click(
            fn=train_rl_agent,
            inputs=[symbol, interval, limit, algorithm, timesteps, learning_rate],
            outputs=[report, download, gr.State()],
        )

        gr.Markdown("""
---
### ℹ️ About this Space

- **Source code**: [github.com/salah55t/crypto-signal-bot](https://github.com/salah55t/crypto-signal-bot)
- **Documentation**: [docs/RL_TRADING.md](https://github.com/salah55t/crypto-signal-bot/blob/main/docs/RL_TRADING.md)
- **Binance data endpoint**: `data-api.binance.vision` (works in US, EU, Asia — no API key required)
- **RL framework**: stable-baselines3 + Gymnasium
- **License**: MIT

⚠️ **Disclaimer**: This is for educational purposes only. Not financial advice.
Crypto trading involves significant risk. Always test with paper trading first.
""")

    return app


if __name__ == "__main__":
    app = create_ui()
    app.launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", 7860)))
