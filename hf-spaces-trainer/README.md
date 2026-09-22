---
title: Crypto RL Trainer
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "5.29.0"
app_file: app.py
pinned: false
license: mit
---

# 🧠 Crypto RL Trainer

Train Reinforcement Learning agents (DQN, PPO, A2C) for cryptocurrency trading on Binance Spot.

## Features

- ✅ Train 3 RL algorithms: DQN, PPO, A2C
- ✅ Works from any region (uses Binance's public data endpoint `data-api.binance.vision`)
- ✅ No Binance API key required for training
- ✅ Custom crypto trading environment with 10 features
- ✅ Real-time progress tracking
- ✅ Download trained models for use in your bot

## How to Use

1. Select a trading pair (e.g., BTCUSDT)
2. Choose timeframe (15m, 1h, 4h, 1d)
3. Set number of bars (default 2000)
4. Choose algorithm: DQN, PPO, or A2C
5. Set training timesteps (50K-200K)
6. Click **Start Training**
7. Download the .zip model file
8. Repeat for other algorithms

## How to Use Trained Models

After downloading your models (`dqn_crypto.zip`, `ppo_crypto.zip`, `a2c_crypto.zip`):

1. Clone the main repository:
   ```bash
   git clone https://github.com/salah55t/crypto-signal-bot.git
   cd crypto-signal-bot
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-rl.txt  # for RL
   ```

3. Place models in `data/models/`:
   ```bash
   mkdir -p data/models
   cp ~/Downloads/dqn_crypto.zip data/models/
   cp ~/Downloads/ppo_crypto.zip data/models/
   cp ~/Downloads/a2c_crypto.zip data/models/
   ```

4. Enable RL in `.env`:
   ```bash
   cp .env.example .env
   echo "ENABLE_RL_STRATEGY=true" >> .env
   ```

5. Run the bot:
   ```bash
   python scripts/run_bot.py --once
   ```

## Documentation

- Main repo: [github.com/salah55t/crypto-signal-bot](https://github.com/salah55t/crypto-signal-bot)
- RL guide: [docs/RL_TRADING.md](https://github.com/salah55t/crypto-signal-bot/blob/main/docs/RL_TRADING.md)
- Render deployment: [docs/RENDER_DEPLOYMENT.md](https://github.com/salah55t/crypto-signal-bot/blob/main/docs/RENDER_DEPLOYMENT.md)

## Architecture

```
This Space ──── fetch candles ────▶ data-api.binance.vision (Binance public)
     │
     ├─ create CryptoTradingEnv (10 features, 3 actions: HOLD/BUY/SELL)
     │
     ├─ train DQN/PPO/A2C (stable-baselines3)
     │
     └─ save .zip model ───▶ user downloads ───▶ data/models/ ───▶ RLStrategy
```

## Disclaimer

⚠️ For educational purposes only. Not financial advice. Crypto trading involves significant risk.

## License

MIT
