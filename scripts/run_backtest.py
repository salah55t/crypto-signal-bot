"""
Backtest runner - tests the strategy on historical data
for one or more symbols.
"""
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.core.data_fetcher import data_fetcher
from src.backtesting.backtester import backtester
from src.utils.logger import log


def main():
    parser = argparse.ArgumentParser(description="Backtest the crypto signal bot")
    parser.add_argument("--symbol", default="BTCUSDT",
                        help="Symbol to backtest (default: BTCUSDT)")
    parser.add_argument("--interval", default="1h",
                        help="Timeframe (default: 1h)")
    parser.add_argument("--limit", type=int, default=1000,
                        help="Number of bars (default: 1000)")
    parser.add_argument("--window", type=int, default=100,
                        help="Analysis window size (default: 100)")
    parser.add_argument("--step", type=int, default=5,
                        help="Step between iterations (default: 5)")
    parser.add_argument("--capital", type=float, default=10000,
                        help="Initial capital (default: $10,000)")
    args = parser.parse_args()

    log.info(f"Fetching {args.limit} {args.interval} candles for {args.symbol}...")
    df = data_fetcher.get_candles(args.symbol, args.interval, limit=args.limit)
    if df.empty:
        log.error("No data fetched")
        return

    result = backtester.run(
        df, args.symbol,
        window=args.window, step=args.step,
        warmup=100,
    )

    if result.get("status") != "success":
        log.error(f"Backtest failed: {result.get('reason')}")
        return

    # Save report
    output_file = Path("data/backtest_report.json")
    backtester.save_report(result, output_file)

    # Print summary
    print("\n" + "=" * 50)
    print(f"  BACKTEST RESULT: {args.symbol}")
    print("=" * 50)
    print(f"  Total Return:     {result['total_return_pct']:+.2f}%")
    print(f"  Buy & Hold:       {result.get('buy_hold_return_pct', 0):+.2f}%")
    print(f"  Final Capital:    ${result['final_capital']:,.2f}")
    print(f"  Total Trades:     {result['total_trades']}")
    print(f"  Wins / Losses:    {result['wins']} / {result['losses']}")
    print(f"  Win Rate:         {result['win_rate_pct']:.1f}%")
    print(f"  Profit Factor:    {result['profit_factor']:.2f}")
    print(f"  Payoff Ratio:     {result.get('payoff_ratio', 0):.2f}")
    print(f"  Expectancy:       {result.get('expectancy_pct_per_trade', 0):+.3f}% / trade")
    print(f"  Max Drawdown:     {result['max_drawdown_pct']:.2f}%")
    print(f"  Fees:             {result.get('fee_pct_per_side', 0.1):.3f}% per side")
    print("=" * 50)


if __name__ == "__main__":
    main()
