"""
Quick test script - runs a single analysis cycle on one symbol
without going through the scheduler. Useful for debugging.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.core.data_fetcher import data_fetcher
from src.analysis.scorer import scorer
from src.utils.logger import log
from src.utils.helpers import fmt_price, fmt_pct
from rich.table import Table
from rich.console import Console


console = Console()


def main():
    symbol = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
    interval = sys.argv[2] if len(sys.argv) > 2 else "1h"
    log.info(f"Testing analysis on {symbol} ({interval})")

    df = data_fetcher.get_candles(symbol, interval, limit=200)
    if df.empty:
        log.error("Failed to fetch candles")
        return

    ob = data_fetcher.get_order_book(symbol, limit=20)
    multi_tf = {interval: df}

    result = scorer.analyze_symbol(df, symbol, multi_tf_data=multi_tf, order_book=ob)

    # Pretty print
    console.print("\n[bold cyan]ANALYSIS RESULT[/]")
    console.print(f"Symbol: {result['symbol']}")
    console.print(f"Direction: [bold]{result['direction']}[/]")
    console.print(f"Confidence: {result['confidence']:.1f}%")
    console.print(f"Weighted Score: {result['weighted_score']:+.1f}")
    console.print(f"Current Price: {fmt_price(result['current_price'])}")
    console.print(f"Expected Rise: {fmt_pct(result['expected_rise_pct'])}")
    console.print(f"Stop Loss: {fmt_price(result['stop_loss'])}")
    console.print(f"Take Profit: {fmt_price(result['take_profit'])}")
    console.print(f"R/R Ratio: {result['risk_reward_ratio']:.2f}:1")

    # Strategy breakdown
    table = Table(title="Strategy Signals")
    table.add_column("Strategy", style="cyan")
    table.add_column("Direction", style="white")
    table.add_column("Score", justify="right")
    table.add_column("Confidence", justify="right")
    table.add_column("Reasons", style="yellow")

    for sig in result["signals"]:
        reasons = "; ".join(sig["reasons"][:2])[:80]
        table.add_row(
            sig["strategy"],
            sig["direction"],
            f"{sig['score']:+.1f}",
            f"{sig['confidence']*100:.1f}%",
            reasons
        )

    console.print(table)


if __name__ == "__main__":
    main()
