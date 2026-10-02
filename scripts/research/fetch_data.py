#!/usr/bin/env python3
"""
v5.28 research - fetch REAL Binance klines for the bot's actual universe.

Universe mirrors production rules (analyzer._fetch_all_usdt_pairs):
  USDT pairs, quoteVolume >= MIN_VOLUME_USDT (5M), volume-sorted, top N=40
  (MAX_SYMBOLS in production is 150; 40 keeps REST weight sane for research
  while covering every liquid name the bot actually signals on).

Data saved OUTSIDE the repo to /home/z/my-project/research_data/ so no
binary artifacts leak into git.

Usage:  python3 scripts/research/fetch_data.py [--top 40] [--bars-4h 1500]
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

BASE = "https://data-api.binance.vision/api/v3"  # public market-data endpoint
OUT_ROOT = Path("/home/z/my-project/research_data")

LEVERAGED = ("UPUSDT", "DOWNUSDT", "BULLUSDT", "BEARUSDT")
STABLE_BASES = {"USDCUSDT", "FDUSDT", "TUSDUSDT", "USDPUSDT", "EURUSDT",
                "DAIUSDT", "AEURUSDT", "USDTUSDT", "PAXGUSDT"}


def http_json(url: str, retries: int = 4) -> object:
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                return json.loads(r.read().decode())
        except Exception as e:  # noqa: BLE001
            wait = 1.5 * (attempt + 1)
            print(f"    retry in {wait:.1f}s ({e})", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"failed: {url}")


def fetch_universe(top: int, min_vol: float) -> list:
    tickers = http_json(f"{BASE}/ticker/24hr")
    rows = []
    for t in tickers:
        sym = t["symbol"]
        if not sym.endswith("USDT") or sym.endswith(LEVERAGED):
            continue
        if sym in STABLE_BASES:
            continue
        qv = float(t["quoteVolume"] or 0)
        if qv < min_vol:
            continue
        rows.append((sym, qv))
    rows.sort(key=lambda x: -x[1])
    return [s for s, _ in rows[:top]]


def fetch_klines(symbol: str, interval: str, total: int) -> list:
    """Paginate backwards until `total` candles collected (or listing start)."""
    out = []
    end_time = None
    while len(out) < total:
        limit = min(1000, total - len(out))
        url = (f"{BASE}/klines?symbol={symbol}&interval={interval}"
               f"&limit={limit}" + (f"&endTime={end_time}" if end_time else ""))
        batch = http_json(url)
        if not batch:
            break
        out = batch + out
        end_time = batch[0][0] - 1
        if len(batch) < limit:
            break
        time.sleep(0.12)
    return out


def to_df(rows: list):
    import pandas as pd
    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_base", "taker_quote", "ignore"])
    for c in ("open", "high", "low", "close", "volume", "quote_volume"):
        df[c] = pd.to_numeric(df[c])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.drop_duplicates("open_time").sort_values("open_time")
    return df.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--bars-4h", type=int, default=1500)
    ap.add_argument("--min-vol", type=float, default=5_000_000)
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "4h").mkdir(exist_ok=True)
    (OUT_ROOT / "1h").mkdir(exist_ok=True)

    print("fetching universe ...", flush=True)
    symbols = fetch_universe(args.top, args.min_vol)
    (OUT_ROOT / "universe.json").write_text(json.dumps(
        {"fetched_at_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
         "rules": f"USDT pairs quoteVolume>={args.min_vol:,.0f}, top {args.top}",
         "symbols": symbols}, indent=1))
    print(f"universe ({len(symbols)}): {', '.join(symbols)}", flush=True)

    for i, sym in enumerate(symbols, 1):
        for tf, want in (("4h", args.bars_4h), ("1h", 1000)):
            path = OUT_ROOT / tf / f"{sym}.csv.gz"
            if path.exists():
                continue
            rows = fetch_klines(sym, tf, want)
            if len(rows) < 300:
                print(f"[{i:2d}/{len(symbols)}] {sym} {tf}: only "
                      f"{len(rows)} bars - skip", flush=True)
                continue
            to_df(rows).to_csv(path, index=False, compression="gzip")
            print(f"[{i:2d}/{len(symbols)}] {sym} {tf}: {len(rows)} bars",
                  flush=True)

    print("done.", flush=True)


if __name__ == "__main__":
    sys.exit(main())
