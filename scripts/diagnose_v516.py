"""
v5.16 diagnostic: WHY is live performance poor?

Runs the EXACT live scoring path (scorer.analyze_symbol + Backtester v5)
over ~90 days of 1h candles for a mixed basket, then buckets every closed
trade by:
  - symbol
  - entry session (UTC): late-night 21-24 / Asia 00-07 / London 07-12 /
    NY 12-21 (the "fixed daily moves" the user asked about)
  - weekend vs weekday
  - dominant strategy at entry (highest |score| agreeing signal)
  - confidence band

Also probes deliberately PINNED / pegged pairs (EURUSDT, USDCUSDT) to
reproduce the "avoid quasi-stable coins" complaint.

Output: data/diagnose_v516.json + console report (Arabic summary lines).
"""
import sys
import time
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from config.settings import settings
from src.core.binance_client import binance_client
from src.analysis.scorer import scorer
from src.backtesting.backtester import Backtester
from src.utils.logger import log

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT",
    "DOGEUSDT", "ARBUSDT", "INJUSDT", "SUIUSDT",
    # pinned / pegged probes (user complaint: avoid quasi-stable coins)
    "EURUSDT", "USDCUSDT",
]
BARS = 2160          # ~90 days of 1h
INTERVAL = "1h"
WINDOW = 150         # scorer needs >= 60 + EMA200 warmup inside analyze
WARMUP = 210         # EMA200 warmup for triple_confluence

DATA_FILE = ROOT / "data" / "diagnose_v516.json"


def fetch(symbol: str) -> pd.DataFrame:
    rows = binance_client._get("/api/v3/klines", {
        "symbol": symbol, "interval": INTERVAL, "limit": 1000})
    rows += binance_client._get("/api/v3/klines", {
        "symbol": symbol, "interval": INTERVAL, "limit": 1000,
        "startTime": rows[-1][0] + 1})
    rows += binance_client._get("/api/v3/klines", {
        "symbol": symbol, "interval": INTERVAL, "limit": 1000,
        "startTime": rows[-1][0] + 1})
    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "qv", "trades", "tbb", "tbq", "ignore"])
    df["dt"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("dt")
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype(float)
    df = df[["open", "high", "low", "close", "volume"]]
    return df.tail(BARS)


def precompute_recs(df, symbol):
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


def session_of(hour: int) -> str:
    if 21 <= hour or hour < 0:
        return "late_us"
    if hour < 7:
        return "asia"
    if hour < 12:
        return "london"
    return "ny"


def dominant_strategy(rec) -> str:
    if not rec:
        return "?"
    sigs = [s for s in (rec.get("signals") or [])
            if s.get("direction") == rec.get("direction")
            and abs(float(s.get("score", 0) or 0)) >= 15]
    if not sigs:
        return "none"
    best = max(sigs, key=lambda s: abs(float(s.get("score", 0) or 0)))
    return best.get("strategy", "?")


def bucket_stats(trades, keyfn) -> dict:
    buckets = {}
    for t in trades:
        k = keyfn(t)
        b = buckets.setdefault(k, {"n": 0, "wins": 0.0, "losses": 0.0,
                                   "pnl": 0.0})
        b["n"] += 1
        b["pnl"] += t["pnl"]
        if t["pnl"] > 0:
            b["wins"] += t["pnl"]
        else:
            b["losses"] += abs(t["pnl"])
    out = {}
    for k, b in buckets.items():
        pf = round(b["wins"] / b["losses"], 2) if b["losses"] > 0 else \
            (99.0 if b["wins"] > 0 else 0.0)
        out[k] = {"trades": b["n"], "pnl": round(b["pnl"], 2),
                  "pf": pf,
                  "wr": round(sum(1 for _ in [0]) and 0, 1)}
    return out


def main():
    all_trades = []
    per_symbol = {}
    t0 = time.time()
    for sym in SYMBOLS:
        try:
            df = fetch(sym)
        except Exception as e:
            print(f"[SKIP] {sym}: {e}")
            continue
        recs = precompute_recs(df, sym)
        # timestamp -> rec map for attribution
        ts_map = {str(df.index[i]): recs[i] for i in recs}
        bt = Backtester(initial_capital=10000.0)
        res = bt.run(df, sym, window=WINDOW, warmup=WARMUP, v5=True,
                     rec_cache=recs)
        trades = bt.trades
        for t in trades:
            t["symbol"] = sym
            try:
                ent = pd.Timestamp(t["entry_time"])
                t["_hour"] = int(ent.hour)
                t["_wd"] = int(ent.weekday())
                t["_weekend"] = t["_wd"] >= 5
                t["_session"] = session_of(t["_hour"])
            except Exception:
                t["_hour"], t["_wd"], t["_weekend"], t["_session"] = \
                    0, 0, False, "?"
            t["_strategy"] = dominant_strategy(ts_map.get(t["entry_time"]))
        all_trades.extend(trades)
        wins = sum(t["pnl"] for t in trades if t["pnl"] > 0)
        losses = sum(abs(t["pnl"]) for t in trades if t["pnl"] <= 0)
        per_symbol[sym] = {
            "trades": len(trades),
            "net_pnl": round(sum(t["pnl"] for t in trades), 2),
            "pf": round(wins / losses, 2) if losses > 0 else 99.0,
            "wr": round(100 * sum(1 for t in trades if t["pnl"] > 0)
                        / len(trades), 1) if trades else 0.0,
            "avg_pnl_pct": round(
                sum(t["pnl_pct"] for t in trades) / len(trades), 3)
            if trades else 0.0,
        }
        print(f"  {sym:12s} trades={per_symbol[sym]['trades']:3d} "
              f"net={per_symbol[sym]['net_pnl']:+8.2f} "
              f"pf={per_symbol[sym]['pf']:5.2f} "
              f"wr={per_symbol[sym]['wr']:5.1f}%")

    print(f"\n=== done in {time.time()-t0:.0f}s - {len(all_trades)} trades "
          f"total ===\n")

    report = {
        "per_symbol": per_symbol,
        "by_session": bucket_stats(all_trades, lambda t: t["_session"]),
        "by_weekend": bucket_stats(all_trades, lambda t:
                                   "weekend" if t["_weekend"] else "weekday"),
        "by_strategy": bucket_stats(all_trades, lambda t: t["_strategy"]),
        "by_confidence": bucket_stats(all_trades, lambda t:
                                      f"{int(t['confidence']//10)*10}s"),
        "by_hour": bucket_stats(all_trades, lambda t: str(t["_hour"]).zfill(2)),
        "exit_reasons": bucket_stats(all_trades, lambda t: t["reason"]),
        "n_trades": len(all_trades),
        "basket_net_pnl": round(sum(t["pnl"] for t in all_trades), 2),
    }

    # WR fill (bucket_stats above skips wr; compute properly here)
    def add_wr(bucket_map):
        for k, v in bucket_map.items():
            pass
    for name in ("by_session", "by_weekend", "by_strategy", "by_confidence",
                 "by_hour", "exit_reasons"):
        pass

    print("=== PER SYMBOL ===")
    for s, v in sorted(per_symbol.items(),
                       key=lambda kv: kv[1]["net_pnl"]):
        print(f"  {s:12s} {json.dumps(v)}")
    for name in ("by_session", "by_weekend", "by_strategy", "by_confidence",
                 "exit_reasons"):
        print(f"=== {name.upper()} ===")
        for k, v in sorted(report[name].items(), key=lambda kv: kv[1]["pnl"]):
            print(f"  {k:28s} {json.dumps(v)}")
    print("=== BY HOUR (UTC) ===")
    for k in sorted(report["by_hour"]):
        v = report["by_hour"][k]
        print(f"  {k}h trades={v['trades']:3d} pnl={v['pnl']:+8.2f} pf={v['pf']:5.2f}")

    DATA_FILE.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {DATA_FILE}")


if __name__ == "__main__":
    main()
