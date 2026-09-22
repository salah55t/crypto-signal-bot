"""
Train RL agents (DQN, PPO, A2C) on crypto data from Binance.

Heavy dependency: stable-baselines3 + PyTorch (~500MB)
Install first:
  pip install stable-baselines3[extra] gymnasium torch

Usage:
  python scripts/train_rl_agents.py --symbol BTCUSDT --interval 1h --limit 2000
  python scripts/train_rl_agents.py --symbol ETHUSDT --algorithms PPO,DQN

Output:
  - data/models/dqn_crypto.zip
  - data/models/ppo_crypto.zip
  - data/models/a2c_crypto.zip
  - data/models/rl_training_report.json
"""
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.core.data_fetcher import data_fetcher
from src.ml.rl_trainer import train_agents, check_rl_dependencies
from src.utils.logger import log


def main():
    parser = argparse.ArgumentParser(description="Train RL trading agents on crypto data")
    parser.add_argument("--symbol", default="BTCUSDT",
                        help="Symbol to train on (default: BTCUSDT)")
    parser.add_argument("--interval", default="1h",
                        help="Timeframe (default: 1h)")
    parser.add_argument("--limit", type=int, default=2000,
                        help="Number of bars (default: 2000 = ~3 months on 1h)")
    parser.add_argument("--timesteps", type=int, default=50000,
                        help="Total training timesteps per algorithm (default: 50000)")
    parser.add_argument("--algorithms", default="DQN,PPO,A2C",
                        help="Comma-separated list (default: DQN,PPO,A2C)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  RL Trading Agents Trainer")
    print("=" * 60)
    print(f"  Symbol:       {args.symbol}")
    print(f"  Interval:     {args.interval}")
    print(f"  Bars:         {args.limit}")
    print(f"  Timesteps:    {args.timesteps}")
    print(f"  Algorithms:   {args.algorithms}")
    print("=" * 60 + "\n")

    if not check_rl_dependencies():
        print("\n[ERROR] Missing dependencies. Install:")
        print("  pip install stable-baselines3[extra] gymnasium torch")
        sys.exit(1)

    log.info(f"Fetching {args.limit} {args.interval} candles for {args.symbol}...")
    df = data_fetcher.get_candles(args.symbol, args.interval, limit=args.limit)
    if df.empty:
        log.error("No data fetched")
        return

    log.info(f"Got {len(df)} candles from {df.index[0]} to {df.index[-1]}")

    algorithms = [a.strip().upper() for a in args.algorithms.split(",")]
    report = train_agents(df, symbol=args.symbol,
                          total_timesteps=args.timesteps,
                          algorithms=algorithms)

    print("\n" + "=" * 60)
    print("  TRAINING SUMMARY")
    print("=" * 60)
    for algo, info in report.get("results", {}).items():
        status = info.get("status", "unknown")
        if status == "success":
            print(f"  {algo:6s} ✅  -> {info.get('path')}")
            eval_data = report.get("evaluations", {}).get(algo, {})
            if eval_data:
                print(f"          Return: {eval_data.get('total_return_pct', 0):+.2f}% | "
                      f"Trades: {eval_data.get('total_trades', 0)} | "
                      f"Win rate: {eval_data.get('win_rate', 0)*100:.1f}% | "
                      f"Max DD: {eval_data.get('max_drawdown_pct', 0):.2f}%")
        else:
            print(f"  {algo:6s} ❌  -> {info.get('error', 'unknown error')}")
    print("=" * 60)

    # Instructions
    print("\n📋 Next steps:")
    print("1. Models saved to data/models/{dqn,ppo,a2c}_crypto.zip")
    print("2. Set ENABLE_RL_STRATEGY=true in .env to activate")
    print("3. Restart the bot - RL Ensemble Strategy will load automatically")
    print("4. The strategy uses majority voting across the 3 agents")


if __name__ == "__main__":
    main()
