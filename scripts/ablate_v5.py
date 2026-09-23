"""Feature ablation on BTCUSDT to isolate the v5 regression source."""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import settings
from src.core.data_fetcher import data_fetcher
from src.analysis.scorer import scorer
from src.backtesting.backtester import Backtester
from src.utils.logger import log

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
WINDOW, WARMUP, BARS = 100, 100, 1000


def recs_for(df, symbol):
    cache = {}
    for i in range(WARMUP, len(df) - 1):
        w = df.iloc[i - WINDOW + 1: i + 1]
        if len(w) < WINDOW:
            continue
        try:
            cache[i] = scorer.analyze_symbol(w, symbol)
        except Exception:
            pass
    return cache


def run(tag, df, symbol, cache, **overrides):
    # save originals
    orig = {
        "MIN_HARMONY": settings.MIN_HARMONY,
        "PENDING_ENTRIES_ENABLED": settings.PENDING_ENTRIES_ENABLED,
        "STRUCTURAL_EXITS_ENABLED": settings.STRUCTURAL_EXITS_ENABLED,
        "CHANDELIER_ENABLED": settings.CHANDELIER_ENABLED,
        "PARTIAL_TP_ENABLED": settings.PARTIAL_TP_ENABLED,
        "MAX_TRADE_HOURS": settings.MAX_TRADE_HOURS,
    }
    settings.MIN_HARMONY = overrides.get("MIN_HARMONY", orig["MIN_HARMONY"])
    settings.PENDING_ENTRIES_ENABLED = overrides.get(
        "PENDING_ENTRIES_ENABLED", orig["PENDING_ENTRIES_ENABLED"])
    settings.STRUCTURAL_EXITS_ENABLED = overrides.get(
        "STRUCTURAL_EXITS_ENABLED", orig["STRUCTURAL_EXITS_ENABLED"])
    settings.CHANDELIER_ENABLED = overrides.get(
        "CHANDELIER_ENABLED", orig["CHANDELIER_ENABLED"])
    settings.PARTIAL_TP_ENABLED = overrides.get(
        "PARTIAL_TP_ENABLED", orig["PARTIAL_TP_ENABLED"])
    settings.MAX_TRADE_HOURS = overrides.get(
        "MAX_TRADE_HOURS", orig["MAX_TRADE_HOURS"])
    try:
        bt = Backtester(10000)
        r = bt.run(df, symbol, window=WINDOW, warmup=WARMUP, v5=True,
                   rec_cache=cache)
        print(f"{symbol} {tag:28} trades={r['total_trades']:>3} "
              f"wr={r['win_rate_pct']:>5.1f} pf={r['profit_factor']:>5.2f} "
              f"exp={r['expectancy_pct_per_trade']:>6.3f} "
              f"ret={r['total_return_pct']:>6.2f}% dd={r['max_drawdown_pct']:.2f}% "
              f"exits={r['exit_reasons']}")
        return r
    finally:
        for k, v in orig.items():
            setattr(settings, k, v)


def main():
    for symbol in SYMBOLS:
        df = data_fetcher.get_candles(symbol, "1h", limit=BARS)
        cache = recs_for(df, symbol)
        print(f"\n=== {symbol} ({len(cache)} recs) ===")
        run("full-v5", df, symbol, cache)
        run("no-pending(market entries)", df, symbol, cache,
            PENDING_ENTRIES_ENABLED=False)
        run("no-struct-exits", df, symbol, cache,
            STRUCTURAL_EXITS_ENABLED=False)
        run("no-chandelier", df, symbol, cache, CHANDELIER_ENABLED=False)
        run("no-partial-tp", df, symbol, cache, PARTIAL_TP_ENABLED=False)
        run("harmony-0.35", df, symbol, cache, MIN_HARMONY=0.35)
        run("harmony-0.30+no-pending", df, symbol, cache,
            MIN_HARMONY=0.30, PENDING_ENTRIES_ENABLED=False)


if __name__ == "__main__":
    main()
