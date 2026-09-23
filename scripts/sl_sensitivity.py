"""
SL sensitivity sweep: chandelier activation threshold x ATR multiple,
using the code's exact engine on identical cached signals.

Motivated by comprehensive_backtest findings:
  - SL_initial trades gave back avg +2.11% MFE before dying at -1.01%
  - winners' MAE is shallow (avg -1.45%) -> initial SL distance is fine;
    the leak is profit protection, not entry placement.

Grid:
  activate_pct x atr_mult = {1.0,0.7,0.5,0.3} x {2.5,2.0,1.5}
Only configs that keep the baseline spirit are reported; C0 is the
current live default (1.0 / 2.5).
"""
import sys
import time
from pathlib import Path
from multiprocessing import Pool

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from config.settings import settings
from src.core.binance_client import binance_client
from src.core.data_fetcher import DataFetcher
from src.analysis.scorer import scorer
from src.backtesting.backtester import Backtester

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT",
           "DOGEUSDT", "ARBUSDT", "INJUSDT", "SUIUSDT"]
BARS, INTERVAL, WINDOW, WARMUP = 3000, "1h", 100, 100

GRID = [(1.0, 2.5), (0.7, 2.5), (0.5, 2.5), (0.5, 2.0),
        (0.5, 1.5), (0.7, 2.0), (0.3, 2.0)]


def fetch_paged(symbol, interval, total):
    end_ms = int(time.time() * 1000)
    chunks, remaining = [], total
    while remaining > 0:
        lim = min(1000, remaining)
        raw = binance_client.get_klines(symbol, interval, limit=lim,
                                        end_time=end_ms)
        if not raw:
            break
        chunks.extend(raw)
        end_ms = int(raw[0][0]) - 1
        remaining -= len(raw)
        if len(raw) < lim:
            break
    df = DataFetcher.klines_to_df(chunks)
    if df.empty:
        return df
    return df[~df.index.duplicated(keep="last")].sort_index()


def run_symbol(symbol):
    t0 = time.time()
    df = fetch_paged(symbol, INTERVAL, BARS)
    if df is None or len(df) < WARMUP + WINDOW:
        return {"symbol": symbol, "rows": []}
    cache = {}
    for i in range(WARMUP, len(df) - 1):
        w = df.iloc[i - WINDOW + 1: i + 1]
        if len(w) < WINDOW:
            continue
        try:
            cache[i] = scorer.analyze_symbol(w, symbol)
        except Exception:
            pass

    idx = df.index
    hi = df["high"].values
    rows = []
    for act, mult in GRID:
        settings.CHANDELIER_ACTIVATE_PCT = act
        settings.CHANDELIER_ATR_MULT = mult
        bt = Backtester(10000.0)
        res = bt.run(df, symbol, window=WINDOW, warmup=WARMUP,
                     v5=True, rec_cache=cache)
        trades = res["trades"]
        # recompute gave-back on SL_initial trades
        gb = []
        for t in trades:
            if t.get("reason") == "SL" and t.get("partials", 0) == 0:
                try:
                    e = idx.searchsorted(pd.Timestamp(t["entry_time"]))
                    x = idx.searchsorted(pd.Timestamp(t["exit_time"]))
                    if x <= e:
                        x = e + 1
                    if 0 <= e < x <= len(idx):
                        mfe = (hi[e:x].max() - t["entry_price"]) / t["entry_price"] * 100
                        gb.append(mfe - t["pnl_pct"])
                except Exception:
                    pass
        rows.append({
            "symbol": symbol, "activate_pct": act, "atr_mult": mult,
            "trades": res["total_trades"],
            "wr": round(res["win_rate_pct"], 1),
            "pf": round(res["profit_factor"], 2),
            "exp": round(res["expectancy_pct_per_trade"], 3),
            "ret": round(res["total_return_pct"], 2),
            "dd": round(res["max_drawdown_pct"], 2),
            "sl_initial": len(gb),
            "gave_back_avg": round(float(np.mean(gb)), 2) if gb else None,
        })
    print(f"[OK] {symbol}: {len(rows)} configs in {time.time()-t0:.0f}s",
          flush=True)
    return {"symbol": symbol, "rows": rows}


def main():
    t0 = time.time()
    with Pool(2) as pool:
        out = pool.map(run_symbol, SYMBOLS)
    rows = [r for o in out for r in o["rows"]]
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "data" / "sl_sensitivity.csv", index=False)

    print("\n" + "=" * 100)
    print(f"{'activate':>8} {'mult':>5} {'trades':>7} {'WR%':>6} {'PF':>6} "
          f"{'Exp%':>7} {'Ret%':>7} {'DD%':>6} {'SLinit':>7} {'gaveBack%':>10}")
    print("-" * 100)
    base = None
    for (act, mult), g in df.groupby(["activate_pct", "atr_mult"], sort=False):
        line = (f"{act:>8} {mult:>5} {g.trades.sum():>7} "
                f"{g.apply(lambda r: r.wr * r.trades, axis=1).sum()/g.trades.sum():>6.1f} ")
        gl = g.apply(lambda r: abs(r.exp) * r.trades, axis=1)  # placeholder
        # weighted PF recompute is complex; use mean PF + mean exp instead
        line += (f"{g.pf.mean():>6.2f} {g.exp.mean():>7.3f} {g.ret.mean():>7.2f} "
                 f"{g.dd.mean():>6.2f} {g.sl_initial.sum():>7} "
                 f"{g.gave_back_avg.mean():>10.2f}")
        if base is None:
            base = line
        print(line + ("   <- current default" if (act, mult) == (1.0, 2.5) else ""))
    print("=" * 100)
    print(f"Done in {time.time()-t0:.0f}s -> data/sl_sensitivity.csv")


if __name__ == "__main__":
    main()
