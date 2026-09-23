"""
A/B backtest comparison: v4.1 baseline vs v5 veteran engine.

Fetches real klines once, precomputes per-bar recommendations ONCE
(the expensive part), then simulates both engines on identical signals:
  - v5=False : old behaviour (market entries, full SL/TP closes)
  - v5=True  : partial TP + chandelier + structure exits + time stop +
               pending limit entries + harmony/volatility gates
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from config.settings import settings
from src.core.data_fetcher import data_fetcher
from src.analysis.scorer import scorer
from src.backtesting.backtester import Backtester
from src.utils.logger import log

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
BARS = 1000
INTERVAL = "1h"
WINDOW = 100
WARMUP = 100


def precompute_recs(df: pd.DataFrame, symbol: str) -> dict:
    """Run the scorer once per bar (cacheable across A/B runs)."""
    cache = {}
    start = WARMUP
    end = len(df) - 1
    t0 = time.time()
    for i in range(start, end):
        window_df = df.iloc[i - WINDOW + 1: i + 1]
        if len(window_df) < WINDOW:
            continue
        try:
            cache[i] = scorer.analyze_symbol(window_df, symbol)
        except Exception as e:
            log.debug(f"rec error bar {i}: {e}")
    log.info(f"[cyan]Precomputed {len(cache)} recs for {symbol} "
             f"in {time.time()-t0:.0f}s")
    return cache


def summarize(tag: str, results: list) -> dict:
    n = len(results)
    trades = sum(r["total_trades"] for r in results)
    wr = [r["win_rate_pct"] for r in results]
    pf = [r["profit_factor"] for r in results]
    exp = [r["expectancy_pct_per_trade"] for r in results]
    dd = [r["max_drawdown_pct"] for r in results]
    ret = [r["total_return_pct"] for r in results]
    row = {
        "engine": tag,
        "trades": trades,
        "avg_wr_pct": round(sum(wr) / n, 1) if n else 0,
        "avg_pf": round(sum(pf) / n, 2) if n else 0,
        "avg_expectancy_pct": round(sum(exp) / n, 3) if n else 0,
        "avg_max_dd_pct": round(sum(dd) / n, 2) if n else 0,
        "avg_return_pct": round(sum(ret) / n, 2) if n else 0,
    }
    return row


def main():
    rows = []
    details = {}
    for symbol in SYMBOLS:
        log.info(f"[bold]=== {symbol} ===[/]")
        df = data_fetcher.get_candles(symbol, INTERVAL, limit=BARS)
        if df is None or len(df) < WARMUP + WINDOW:
            log.error(f"Insufficient data for {symbol}")
            continue
        cache = precompute_recs(df, symbol)

        bt_base = Backtester(initial_capital=10000)
        base = bt_base.run(df, symbol, window=WINDOW, warmup=WARMUP,
                           v5=False, rec_cache=cache)

        bt_v5 = Backtester(initial_capital=10000)
        v5 = bt_v5.run(df, symbol, window=WINDOW, warmup=WARMUP,
                       v5=True, rec_cache=cache)

        rows.append(summarize("v4.1", [base]))
        rows.append(summarize("v5", [v5]))
        details[symbol] = {
            "baseline": {k: v for k, v in base.items()
                         if k not in ("equity_curve", "trades")},
            "v5": {k: v for k, v in v5.items()
                   if k not in ("equity_curve", "trades")},
            "v5_exit_reasons": v5.get("exit_reasons", {}),
            "v5_partial_trades": v5.get("partial_tp_trades", 0),
        }

    # summary table
    print("\n" + "=" * 88)
    print(f"{'engine':8} {'trades':>7} {'avgWR%':>8} {'avgPF':>7} "
          f"{'Exp%/trade':>11} {'avgDD%':>8} {'avgRet%':>9}")
    print("-" * 88)
    for r in rows:
        print(f"{r['engine']:8} {r['trades']:>7} {r['avg_wr_pct']:>8} "
              f"{r['avg_pf']:>7} {r['avg_expectancy_pct']:>11} "
              f"{r['avg_max_dd_pct']:>8} {r['avg_return_pct']:>9}")
    print("=" * 88)

    out = Path(ROOT) / "data" / "v5_ab_comparison.json"
    import json
    out.write_text(json.dumps({"rows": rows, "details": details},
                              indent=2, default=str))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
