"""
Comprehensive backtest using the bot's EXACT live logic (v5 engine),
scaled to the local PC capacity (2 cores / 3GB RAM).

Focus (per user request): STOP-LOSS logic in depth.
  - Initial SL placement (from the code's SL engine)
  - SL evolution: TP1 partial -> BE+fees move, chandelier 2.5xATR trail
  - SL exit outcomes: full-risk SL / BE-protected SL / trailed-profit SL
  - SL quality: hit rate, avg risk per SL, MFE before SL (missed profit)

Engine runs BOTH:
  v5=False : v4.1 baseline (plain SL/TP closes)   <- A/B reference
  v5=True  : veteran engine (the live code path)

Data: paginated klines (1000/request) directly from binance_client.
  1h  x 3000 bars (~125 days) x 18 symbols
  15m x 2000 bars (~21 days)  x 8 symbols  (production timeframe)

Outputs:
  data/comprehensive_backtest.json  (aggregates + per-symbol + SL stats)
  data/comprehensive_trades.csv     (every trade, full lifecycle fields)
"""
import sys
import time
import json
import traceback
from pathlib import Path
from multiprocessing import Pool

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from config.settings import settings
from src.core.data_fetcher import DataFetcher
from src.core.binance_client import binance_client
from src.analysis.scorer import scorer
from src.backtesting.backtester import Backtester
from src.indicators.technical import atr as ta_atr

# ----------------------------------------------------------------------
SYMBOLS_1H = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT",
    "AVAXUSDT", "DOTUSDT", "LINKUSDT", "UNIUSDT", "AAVEUSDT", "ATOMUSDT",
    "ARBUSDT", "OPUSDT", "APTUSDT", "SUIUSDT", "INJUSDT", "DOGEUSDT",
]
SYMBOLS_15M = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT",
    "DOGEUSDT", "ARBUSDT", "INJUSDT", "SUIUSDT",
]
BARS_1H = 3000      # ~125 days
BARS_15M = 2000     # ~21 days (production TF)
WINDOW = 100
WARMUP = 100
CAPITAL = 10000.0

JOBS = ([(s, "1h", BARS_1H) for s in SYMBOLS_1H]
        + [(s, "15m", BARS_15M) for s in SYMBOLS_15M])


# ----------------------------------------------------------------------
def fetch_paged(symbol: str, interval: str, total: int) -> pd.DataFrame:
    """Paginate klines backwards (1000/request) into one clean DataFrame."""
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
    return (df[~df.index.duplicated(keep="last")].sort_index())


def precompute_recs(df: pd.DataFrame, symbol: str) -> dict:
    """Score every bar once with the live scorer (shared by both engines)."""
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


class InstrumentedBacktester(Backtester):
    """Same engine, richer trade records (no behaviour change)."""
    def _open_position(self, rec, timestamp, entry_price=None):
        pos = super()._open_position(rec, timestamp, entry_price)
        pos["initial_stop_loss"] = pos["stop_loss"]
        pos["initial_atr"] = pos["atr"]
        return pos

    def _finish_trade(self, pos, exit_price, timestamp, reason):
        rec = super()._finish_trade(pos, exit_price, timestamp, reason)
        rec["initial_stop_loss"] = pos.get("initial_stop_loss")
        rec["final_stop_loss"] = pos["stop_loss"]
        rec["peak_price"] = pos["peak_price"]
        rec["trough_price"] = pos["trough_price"]
        rec["initial_atr"] = pos.get("initial_atr", 0.0)
        return rec


def classify(t: dict) -> str:
    """SL-outcome classification from the code's own lifecycle fields."""
    r = t.get("reason", "?")
    if r == "SL":
        if t.get("partials", 0) == 0:
            return "SL_initial"          # full risk, never reached TP1
        if t.get("pnl_pct", 0) >= 0:
            return "SL_protected"        # TP1 banked, BE/trail stopped rest
        return "SL_partial_loss"         # TP1 banked but remainder under BE
    return r


def add_mfe_mae(trades: list, df: pd.DataFrame) -> None:
    """MFE/MAE in % from real highs/lows between entry and exit (longs)."""
    idx = df.index
    hi, lo = df["high"].values, df["low"].values
    for t in trades:
        try:
            e = idx.searchsorted(pd.Timestamp(t["entry_time"]))
            x = idx.searchsorted(pd.Timestamp(t["exit_time"]))
            if x <= e:
                x = e + 1
            if e < 0 or x > len(idx):
                continue
            entry = t["entry_price"]
            t["mfe_pct"] = (hi[e:x].max() - entry) / entry * 100
            t["mae_pct"] = (lo[e:x].min() - entry) / entry * 100
        except Exception:
            t["mfe_pct"], t["mae_pct"] = None, None


def run_job(job):
    symbol, interval, bars = job
    t0 = time.time()
    tag = f"{symbol}-{interval}"
    try:
        df = fetch_paged(symbol, interval, bars)
        if df is None or len(df) < WARMUP + WINDOW:
            return {"tag": tag, "status": "skip", "reason": "insufficient data"}

        cache = precompute_recs(df, symbol)

        base = InstrumentedBacktester(CAPITAL).run(
            df, symbol, window=WINDOW, warmup=WARMUP, v5=False, rec_cache=cache)
        v5 = InstrumentedBacktester(CAPITAL).run(
            df, symbol, window=WINDOW, warmup=WARMUP, v5=True, rec_cache=cache)

        add_mfe_mae(v5["trades"], df)
        for t in v5["trades"]:
            t["class"] = classify(t)
            ini, fin = t.get("initial_stop_loss"), t.get("final_stop_loss")
            t["sl_moved"] = bool(ini is not None and fin is not None
                                 and abs(fin - ini) > 1e-12)
            if t.get("initial_atr"):
                t["sl_dist_atr"] = (t["entry_price"] - (ini or 0)) / t["initial_atr"]
                t["sl_dist_pct"] = (t["entry_price"] - (ini or 0)) / t["entry_price"] * 100

        # SL placement stats from the code's gated recs
        sl_pct, sl_atr = [], []
        for rec in cache.values():
            if rec and rec.get("direction") == "bullish" \
                    and not rec.get("decision", {}).get("vetoed", False) \
                    and rec.get("admission_confidence",
                                rec.get("confidence", 0)) >= settings.MIN_CONFIDENCE \
                    and rec.get("risk_reward_ratio", 0) >= settings.MIN_RR_RATIO:
                e, s = rec.get("entry_price") or rec.get("current_price"), rec.get("stop_loss")
                a = rec.get("atr")
                if e and s and e > 0:
                    sl_pct.append((e - s) / e * 100)
                    if a:
                        sl_atr.append((e - s) / a)

        strip = ("equity_curve", "trades")
        out = {
            "tag": tag, "symbol": symbol, "interval": interval,
            "bars": len(df), "recs": len(cache),
            "status": "success",
            "baseline": {k: v for k, v in base.items() if k not in strip},
            "v5": {k: v for k, v in v5.items() if k not in strip},
            "v5_trades": v5["trades"],
            "v5_class_counts": pd.Series(
                [t["class"] for t in v5["trades"]]).value_counts().to_dict(),
            "sl_placement": {
                "gated_recs": len(sl_pct),
                "sl_dist_pct_avg": round(float(np.mean(sl_pct)), 3) if sl_pct else None,
                "sl_dist_pct_p90": round(float(np.percentile(sl_pct, 90)), 3) if sl_pct else None,
                "sl_dist_atr_avg": round(float(np.mean(sl_atr)), 3) if sl_atr else None,
            },
            "seconds": round(time.time() - t0, 1),
        }
        print(f"[OK] {tag}: bars={len(df)} trades(v5)={v5['total_trades']} "
              f"PF={v5['profit_factor']:.2f} | {out['seconds']}s", flush=True)
        return out
    except Exception as e:
        print(f"[FAIL] {tag}: {e}", flush=True)
        traceback.print_exc()
        return {"tag": tag, "status": "failed", "reason": str(e)}


# ----------------------------------------------------------------------
def agg(results, key):
    """Weighted aggregate over successful runs."""
    ok = [r for r in results if r.get("status") == "success"]
    if not ok:
        return {}
    trades = sum(r[key]["total_trades"] for r in ok)
    wins = sum(r[key]["wins"] for r in ok)
    gw = sum(r[key]["avg_win"] * r[key]["wins"] for r in ok)
    gl = sum(abs(r[key]["avg_loss"]) * r[key]["losses"] for r in ok)
    pf = gw / gl if gl > 0 else 0.0
    exp = [r[key]["expectancy_pct_per_trade"] for r in ok]
    dd = [r[key]["max_drawdown_pct"] for r in ok]
    ret = [r[key]["total_return_pct"] for r in ok]
    return {
        "runs": len(ok), "trades": trades,
        "win_rate_pct": round(wins / trades * 100, 1) if trades else 0,
        "profit_factor": round(pf, 2),
        "expectancy_pct_per_trade": round(float(np.mean(exp)), 3),
        "avg_max_dd_pct": round(float(np.mean(dd)), 2),
        "avg_return_pct": round(float(np.mean(ret)), 2),
    }


def main():
    t0 = time.time()
    print(f"Comprehensive backtest: {len(JOBS)} jobs "
          f"({len(SYMBOLS_1H)}x1h/{BARS_1H}bars + {len(SYMBOLS_15M)}x15m/{BARS_15M}bars)",
          flush=True)
    with Pool(2) as pool:
        results = pool.map(run_job, JOBS)

    ok = [r for r in results if r.get("status") == "success"]
    print(f"\nDone: {len(ok)}/{len(JOBS)} jobs in {time.time()-t0:.0f}s", flush=True)

    # ---- all trades CSV (v5) ----
    rows = []
    for r in ok:
        for t in r["v5_trades"]:
            rows.append({
                "symbol": r["symbol"], "tf": r["interval"],
                "entry_time": t["entry_time"], "exit_time": t["exit_time"],
                "entry": t["entry_price"], "exit": t["exit_price"],
                "pnl": round(t["pnl"], 4), "pnl_pct": round(t["pnl_pct"], 3),
                "reason": t["reason"], "class": t["class"],
                "partials": t["partials"], "bars_held": t["bars_held"],
                "confidence": t["confidence"], "sl_moved": t["sl_moved"],
                "sl_dist_pct": t.get("sl_dist_pct"),
                "sl_dist_atr": t.get("sl_dist_atr"),
                "mfe_pct": t.get("mfe_pct"), "mae_pct": t.get("mae_pct"),
            })
    csv_path = ROOT / "data" / "comprehensive_trades.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    # ---- exit class breakdown (all v5 trades) ----
    cls_counts, cls_pnl = {}, {}
    for r in ok:
        for t in r["v5_trades"]:
            c = t["class"]
            cls_counts[c] = cls_counts.get(c, 0) + 1
            cls_pnl.setdefault(c, []).append(t["pnl_pct"])
    class_table = {
        c: {"count": n,
            "avg_pnl_pct": round(float(np.mean(cls_pnl[c])), 3),
            "share_pct": None}
        for c, n in cls_counts.items()
    }
    tot = sum(cls_counts.values())
    for c in class_table:
        class_table[c]["share_pct"] = round(cls_counts[c] / tot * 100, 1) if tot else 0

    # ---- SL deep-dive ----
    sl_rows = [t for r in ok for t in r["v5_trades"] if t["class"] == "SL_initial"]
    be_rows = [t for r in ok for t in r["v5_trades"] if t["class"] == "SL_protected"]
    tr_rows = [t for r in ok for t in r["v5_trades"]
               if t["class"] == "SL_initial" and t.get("mfe_pct") is not None]
    sl_stats = {
        "sl_initial_count": len(sl_rows),
        "sl_initial_share_pct": round(len(sl_rows) / tot * 100, 1) if tot else 0,
        "sl_initial_avg_loss_pct": round(float(np.mean(
            [t["pnl_pct"] for t in sl_rows])), 3) if sl_rows else None,
        "sl_initial_avg_bars_held": round(float(np.mean(
            [t["bars_held"] for t in sl_rows])), 1) if sl_rows else None,
        "sl_initial_mfe_before_exit_avg_pct": round(float(np.mean(
            [t["mfe_pct"] for t in tr_rows])), 3) if tr_rows else None,
        "sl_protected_count": len(be_rows),
        "sl_protected_avg_pnl_pct": round(float(np.mean(
            [t["pnl_pct"] for t in be_rows])), 3) if be_rows else None,
        "sl_placement_gated_recs": int(np.sum(
            [r["sl_placement"]["gated_recs"] for r in ok])),
        "sl_dist_pct_avg": round(float(np.mean([
            r["sl_placement"]["sl_dist_pct_avg"] for r in ok
            if r["sl_placement"]["sl_dist_pct_avg"] is not None])), 3),
        "sl_dist_atr_avg": round(float(np.mean([
            r["sl_placement"]["sl_dist_atr_avg"] for r in ok
            if r["sl_placement"]["sl_dist_atr_avg"] is not None])), 3),
    }

    # ---- A/B aggregates ----
    summary = {
        "config": {"symbols_1h": SYMBOLS_1H, "symbols_15m": SYMBOLS_15M,
                   "bars_1h": BARS_1H, "bars_15m": BARS_15M,
                   "window": WINDOW, "warmup": WARMUP, "capital": CAPITAL,
                   "min_confidence": settings.MIN_CONFIDENCE,
                   "min_rr": settings.MIN_RR_RATIO,
                   "min_harmony": settings.MIN_HARMONY,
                   "fee_pct": settings.TRADING_FEE_PCT},
        "aggregate_all": {"baseline": agg(results, "baseline"),
                          "v5": agg(results, "v5")},
        "aggregate_1h_only": {
            "baseline": agg([r for r in ok if r["interval"] == "1h"], "baseline"),
            "v5": agg([r for r in ok if r["interval"] == "1h"], "v5")},
        "aggregate_15m_only": {
            "baseline": agg([r for r in ok if r["interval"] == "15m"], "baseline"),
            "v5": agg([r for r in ok if r["interval"] == "15m"], "v5")},
        "exit_class_table": class_table,
        "sl_stats": sl_stats,
        "per_symbol": [{
            "symbol": r["symbol"], "interval": r["interval"], "bars": r["bars"],
            "v5_trades": r["v5"]["total_trades"],
            "v5_wr": r["v5"]["win_rate_pct"], "v5_pf": r["v5"]["profit_factor"],
            "v5_exp": r["v5"]["expectancy_pct_per_trade"],
            "v5_dd": r["v5"]["max_drawdown_pct"],
            "v5_ret": r["v5"]["total_return_pct"],
            "base_trades": r["baseline"]["total_trades"],
            "base_pf": r["baseline"]["profit_factor"],
            "base_ret": r["baseline"]["total_return_pct"],
            "buy_hold_ret": r["v5"]["buy_hold_return_pct"],
            "class_counts": r["v5_class_counts"],
        } for r in ok],
    }

    json_path = ROOT / "data" / "comprehensive_backtest.json"
    json_path.write_text(json.dumps(summary, indent=2, default=str))

    # ---- console tables ----
    print("\n" + "=" * 96)
    print("AGGREGATE (weighted over symbols)")
    for scope in ("aggregate_all", "aggregate_1h_only", "aggregate_15m_only"):
        s = summary[scope]
        print(f"\n-- {scope} --")
        for eng in ("baseline", "v5"):
            a = s[eng]
            if a:
                print(f"  {eng:8} trades={a['trades']:>4} WR={a['win_rate_pct']:>5}% "
                      f"PF={a['profit_factor']:>5} Exp={a['expectancy_pct_per_trade']:>6}%/trade "
                      f"DD={a['avg_max_dd_pct']:>5}% Ret={a['avg_return_pct']:>7}%")
    print("\nEXIT CLASSES (v5):")
    for c, d in sorted(class_table.items(), key=lambda kv: -kv[1]["count"]):
        print(f"  {c:18} n={d['count']:>4} ({d['share_pct']:>5}%) avgPnL={d['avg_pnl_pct']:>7}%")
    print("\nSL DEEP-DIVE:")
    for k, v in sl_stats.items():
        print(f"  {k:38} {v}")
    print(f"\nSaved: {json_path}\nSaved: {csv_path}")


if __name__ == "__main__":
    main()
