"""
Train ML model (Random Forest) on historical data.
Saves model to data/models/rf_direction.pkl for use by MLStrategy.
"""
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.core.data_fetcher import data_fetcher
from src.strategies.ml_strategy import MLStrategy
from src.utils.logger import log
from src.utils.helpers import save_json, to_json_safe


def main():
    parser = argparse.ArgumentParser(description="Train ML model for crypto signal bot")
    parser.add_argument("--symbol", default="BTCUSDT",
                        help="Symbol to train on (default: BTCUSDT)")
    parser.add_argument("--interval", default="1h",
                        help="Timeframe (default: 1h)")
    parser.add_argument("--limit", type=int, default=1000,
                        help="Number of bars (default: 1000)")
    args = parser.parse_args()

    log.info(f"Fetching {args.limit} {args.interval} candles for {args.symbol}...")
    df = data_fetcher.get_candles(args.symbol, args.interval, limit=args.limit)
    if df.empty:
        log.error("No data fetched")
        return

    ml = MLStrategy()
    report = ml.train(df, symbol_for_log=args.symbol)

    if report.get("status") != "success":
        log.error(f"Training failed: {report.get('reason')}")
        return

    # Save training report
    out_path = Path("data/ml_training_report.json")
    save_json(to_json_safe(report), out_path)
    log.info(f"Training report saved to {out_path}")

    print("\n" + "=" * 50)
    print(f"  ML TRAINING REPORT: {args.symbol}")
    print("=" * 50)
    print(f"  Status:           {report['status']}")
    print(f"  Accuracy:         {report['accuracy']:.4f}")
    print(f"  Total Samples:    {report['samples']}")
    print(f"  Train Samples:    {report['train_samples']}")
    print(f"  Test Samples:     {report['test_samples']}")
    print("\nTop 5 Features:")
    for name, importance in list(report["feature_importance"].items())[:5]:
        print(f"  - {name:20s}: {importance:.4f}")
    print("=" * 50)


if __name__ == "__main__":
    main()
