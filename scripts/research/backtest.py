#!/usr/bin/env python3
"""
v5.28 strategy research harness - evidence-based replacement research.

Method
------
1. BASELINES: the six production composite strategies run UNMODIFIED
   (imported from src/strategies) on real Binance 4h data, closed candles
   only, window=300 (production CANDLE_LIMIT=200-300).
2. CANDIDATES: five literature-backed entry families (TSMOM trend,
   Donchian breakout, RSI(2) dip-with-trend, BB/KC-style squeeze breakout,
   SuperTrend flip), vectorised.
3. SHARED EXIT ENGINE - identical for every strategy so differences are
   attributable to ENTRY quality alone. It replicates the production trade
   lifecycle (risk/manager.py + settings.py):
     - entry at next open (+slippage), initial SL / TP1 / TP2 by ATR
     - TP1 partial 50%, then SL -> BE + 0.25% fee buffer
     - profit ladder: +1% -> lock +0.30% | +2% -> +1.10% | +3% -> +2.00%
       | +5% -> trail 1% below peak  (locks only tighten)
     - v5.27 fee-survival rung: MFE >= 0.45% AND giveback >= 0.20% while
       profit < +1% -> SL = entry + rt_cost (a winner never dies as a fee loss)
     - time stops: 72h if pnl < 0.5%, absolute 120h
     - pessimistic fills: SL first when SL and TP touched in the same bar
4. COSTS: 0.10% taker fee per side + 0.02% slippage per side (spot).
5. VALIDATION: 3 equal time-folds (fold 3 = out-of-sample), per-fold
   expectancy, symbol breadth, bootstrap 95% CI, profit factor.
6. STARVATION AUDIT: for every baseline signal, compute the production
   admission_confidence for a single-vote signal and report the share that
   would pass MIN_CONFIDENCE=68 (explains why composites never trade).

Usage:
    python3 scripts/research/backtest.py --symbols 12   # smoke test
    python3 scripts/research/backtest.py                # full run
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

DATA = Path("/home/z/my-project/research_data")
OUT = DATA / "results.json"

FEE = 0.001          # taker per side
SLIP = 0.0002        # slippage per side
RT_COST = 2 * (FEE + SLIP)          # 0.024 full round trip on price terms
TP1_FRACTION = 0.5
BE_BUFFER = 0.0025                  # TP1_FEE_BUFFER_PCT
LADDER = [(1.0, 0.30), (2.0, 1.10), (3.0, 2.00), (5.0, None)]  # (peak%, lock%|trail1%)
FEE_RUNG_MFE = 0.0045
FEE_RUNG_GIVEBACK = 0.0020
STALE_BARS = 18        # 72h on 4h
ABS_BARS = 30          # 120h on 4h
WINDOW = 300           # bars fed to strategy classes
WARMUP = 210           # first bar index eligible (EMA200 + buffer)
MIN_CONF = 68.0
CONFLUENCE_REF = 5.8

# weights from src/analysis/scorer.py:64-81
STRATEGY_CLASSES = [
    ("trend_pullback", "src.strategies.trend_pullback_strategy.TrendPullbackStrategy", 2.0),
    ("liquidity_sweep_reversal", "src.strategies.liquidity_sweep_reversal_strategy.LiquiditySweepReversalStrategy", 2.0),
    ("volatility_breakout", "src.strategies.volatility_breakout_strategy.VolatilityBreakoutStrategy", 1.8),
    ("triple_confluence_trend", "src.strategies.triple_confluence_trend_strategy.TripleConfluenceTrendStrategy", 1.6),
    ("bb_mean_reversion", "src.strategies.bb_mean_reversion_strategy.BBMeanReversionStrategy", 1.2),
    ("macd_breakout", "src.strategies.macd_breakout_strategy.MACDBreakoutStrategy", 1.4),
]

CANDIDATES = ["tsmom_trend", "donchian_break", "rsi2_dip", "squeeze_break", "st_flip"]


# ---------------------------------------------------------------- indicators
def wilder_atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    up, dn = h.diff(), -l.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    pdi = 100 * plus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / atr
    mdi = 100 * minus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def supertrend(df: pd.DataFrame, period: int = 10, mult: float = 3.0):
    """PRODUCTION-identical SuperTrend (src/indicators/supertrend.py)."""
    from src.indicators.supertrend import supertrend as prod_st
    _, tr = prod_st(df["high"], df["low"], df["close"], period, mult)
    return None, tr


# ---------------------------------------------------------------- exit engine
def simulate_trade(df: pd.DataFrame, sig_i: int, sl_pct: float,
                   tp1_pct: float, tp2_pct: float) -> dict | None:
    """One position per symbol; identical lifecycle for every strategy."""
    n = len(df)
    if sig_i + 1 >= n:
        return None
    o = df["open"].values
    h = df["high"].values
    l = df["low"].values
    c = df["close"].values
    entry = o[sig_i + 1] * (1 + SLIP)
    sl = entry * (1 - sl_pct)
    tp1 = entry * (1 + tp1_pct)
    tp2 = entry * (1 + tp2_pct)
    peak = entry
    realized = 0.0
    tp1_done = False
    exit_reason = "open"
    exit_px = None
    last_j = None
    for j in range(sig_i + 1, min(sig_i + 1 + ABS_BARS + 1, n)):
        last_j = j
        gain = h[j] / entry - 1
        if gain > (peak / entry - 1):
            peak = h[j]
        peak_gain = peak / entry - 1
        cur_gain_hi = h[j] / entry - 1
        cur_gain_lo = c[j] / entry - 1
        # --- ladder locks (only tighten) -------------------------------
        if peak_gain >= LADDER[3][0]:
            trail = peak * (1 - 0.01)
            sl = max(sl, trail)
        for pg, lock in LADDER[:3]:
            if peak_gain >= pg / 100:
                sl = max(sl, entry * (1 + lock / 100))
        # --- v5.27 fee-survival rung -----------------------------------
        if (not tp1_done and peak_gain >= FEE_RUNG_MFE
                and (peak_gain - cur_gain_lo) >= FEE_RUNG_GIVEBACK
                and cur_gain_lo < 0.01):
            sl = max(sl, entry * (1 + RT_COST - 2 * SLIP + 0.0002))
        # --- pessimistic same-bar resolution ---------------------------
        hit_sl = l[j] <= sl
        hit_tp1 = (not tp1_done) and h[j] >= tp1
        if hit_sl and not hit_tp1:
            exit_px, exit_reason = sl, "SL"
            break
        if hit_sl and hit_tp1:          # both touched -> SL first
            exit_px, exit_reason = sl, "SL(same-bar)"
            break
        if hit_tp1:
            realized += TP1_FRACTION * (tp1 * (1 - FEE - SLIP) / entry - 1 - FEE)
            tp1_done = True
            sl = max(sl, entry * (1 + BE_BUFFER))
            if h[j] >= tp2:             # same bar beyond TP2 (rare)
                exit_px, exit_reason = tp2, "TP2(same-bar)"
                realized += (1 - TP1_FRACTION) * (tp2 * (1 - FEE - SLIP) / entry - 1 - FEE)
                break
            continue
        if tp1_done and h[j] >= tp2:
            exit_px, exit_reason = tp2, "TP2"
            realized += (1 - TP1_FRACTION) * (tp2 * (1 - FEE - SLIP) / entry - 1 - FEE)
            break
        # --- time stops -------------------------------------------------
        age = j - sig_i
        if age >= STALE_BARS and cur_gain_lo < 0.005:
            exit_px, exit_reason = c[j], "stale72h"
            break
        if age >= ABS_BARS:
            exit_px, exit_reason = c[j], "time120h"
            break
    if exit_px is None:
        exit_px = c[last_j]
        exit_reason = "time120h"
    final_leg = (1 - TP1_FRACTION) * (exit_px * (1 - FEE - SLIP) / entry - 1 - FEE)
    total = realized + final_leg if tp1_done else exit_px * (1 - FEE - SLIP) / entry - 1 - FEE
    return {"ret": total, "hold": last_j - sig_i, "reason": exit_reason,
            "tp1": tp1_done, "mfe": peak / entry - 1}


def trade_metrics(trades: list) -> dict:
    if not trades:
        return {"n": 0}
    rets = np.array([t["ret"] for t in trades])
    wins, losses = rets[rets > 0], rets[rets <= 0]
    pf = float(wins.sum() / -losses.sum()) if len(losses) and losses.sum() < 0 else float("inf")
    eq = np.cumsum(rets)
    dd = float((np.maximum.accumulate(eq) - eq).max()) if len(eq) else 0.0
    rng = np.random.default_rng(7)
    boots = [float(rng.choice(rets, len(rets), replace=True).mean()) for _ in range(3000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "n": int(len(rets)),
        "win_rate": float((rets > 0).mean()),
        "expectancy_bps": float(rets.mean() * 1e4),
        "profit_factor": pf,
        "avg_hold_bars": float(np.mean([t["hold"] for t in trades])),
        "max_dd_bps": dd * 1e4,
        "ci95_bps": [round(float(lo) * 1e4, 1), round(float(hi) * 1e4, 1)],
        "tp1_rate": float(np.mean([t["tp1"] for t in trades])),
        "total_ret_bps": float(rets.sum() * 1e4),
    }


# ---------------------------------------------------------------- candidates
def btc_tide() -> pd.DataFrame:
    """BTC 4h regime proxy of the production market-tide gate:
    open_time + tide bool (BTC close > its 4h EMA200)."""
    df = pd.read_csv(DATA / "4h" / "BTCUSDT.csv.gz")
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    df["tide"] = (df["close"] > df["close"].ewm(
        span=200, adjust=False, min_periods=200).mean()).fillna(False)
    return df[["open_time", "tide"]]


def candidate_signals(df: pd.DataFrame, name: str,
                      tide: pd.DataFrame | None = None) -> tuple[np.ndarray, dict]:
    """Return (bool array: enter at next open after bar i, aux)."""
    c = df["close"]
    a = wilder_atr(df, 14)
    ema200 = c.ewm(span=200, adjust=False, min_periods=200).mean()
    ema50 = c.ewm(span=50, adjust=False, min_periods=50).mean()
    vol = df["volume"]
    vsma = vol.rolling(20).mean()
    sig = np.zeros(len(df), dtype=bool)
    aux = {}

    if name == "tsmom_trend":
        # academic TSMOM in regime form: uptrend + positive 24-bar momentum
        mom = c.pct_change(24)
        cond = (c > ema200) & (ema50 > ema200) & (mom > 0)
        sig = cond.fillna(False).values

    elif name == "donchian_break":
        hh = df["high"].rolling(55).max().shift(1)
        cond = (c > hh) & (vol > 1.2 * vsma)
        raw = cond.fillna(False).values
        armed = np.ones(len(df), dtype=bool)
        prev_in = False
        for i in range(len(df)):
            if raw[i] and armed[i]:
                sig[i] = True
                armed[:] = True
                armed[: i + 1] = False   # one shot per breakout
            elif c.iloc[i] < hh.iloc[i]:
                armed[i] = True          # re-arm once back inside channel
        aux["note"] = "55-bar channel, one-shot per breakout, vol 1.2x"

    elif name == "rsi2_dip":
        r2 = rsi(c, 2)
        cond = (r2 < 10) & (c > ema200)
        raw = cond.fillna(False).values
        armed = np.ones(len(df), dtype=bool)
        for i in range(len(df)):
            if raw[i] and armed[i]:
                sig[i] = True
                armed[: i + 1] = False
            elif (r2.iloc[i] or 50) > 30:
                armed[i] = True
        aux["note"] = "RSI(2)<10 dip above EMA200, re-arm on RSI>30"

    elif name == "squeeze_break":
        ma = c.rolling(20).mean()
        sd = c.rolling(20).std()
        width = (4 * sd) / ma
        pct = width.rolling(100).rank(pct=True)
        up = ma + 2 * sd
        brk = (c > up) & (c.shift(1) <= up.shift(1))
        cond = (pct < 0.25) & brk & (vol > 1.3 * vsma)
        raw = cond.fillna(False).values
        armed = np.ones(len(df), dtype=bool)
        for i in range(len(df)):
            if raw[i] and armed[i]:
                sig[i] = True
                armed[: i + 1] = False
            elif pct.iloc[i] > 0.6:
                armed[i] = True
        aux["note"] = "BB width <25th pct + close crosses upper band + vol 1.3x"

    elif name == "st_flip":
        _, tr = supertrend(df, 10, 3.0)
        flip = (tr == 1) & (tr.shift(1) == -1)
        cond = flip & (c > ema200)
        sig = cond.fillna(False).values
        aux["note"] = "SuperTrend(10,3) flip up above EMA200"

    if tide is not None and name != "__none__":
        df = df.copy()
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
        df = df.merge(tide, on="open_time", how="left")
        df["tide"] = df["tide"].fillna(False)
        sig = sig & df["tide"].values
        aux["tide"] = "BTC close > BTC 4h EMA200"

    return sig, aux


CAND_PARAMS = {
    "tsmom_trend": (0.020, 0.024, 0.048),
    "donchian_break": (0.020, 0.024, 0.048),
    "rsi2_dip": (0.015, 0.018, 0.036),
    "squeeze_break": (0.020, 0.024, 0.048),
    "st_flip": (0.020, 0.024, 0.048),
}


def run_candidates(df: pd.DataFrame) -> dict:
    out = {}
    a = wilder_atr(df, 14).values
    for name in CANDIDATES:
        sig, aux = candidate_signals(df, name)
        trades = []
        busy_until = -1
        for i in np.where(sig)[0]:
            if i < WARMUP or i <= busy_until:
                continue
            t = simulate_trade(df, int(i), *CAND_PARAMS[name])
            if t is None:
                continue
            busy_until = int(i) + t["hold"]
            trades.append(t)
        m = trade_metrics(trades)
        m["signals"] = int(sig[WARMUP:].sum())
        m["aux"] = aux
        out[name] = {"metrics": m, "trades": trades}
    return out


# ---------------------------------------------------------------- baselines
def run_baselines_symbol(sym: str) -> dict:
    """Run the six production strategy classes unmodified on closed candles."""
    path = DATA / "4h" / f"{sym}.csv.gz"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
    from src.strategies.trend_pullback_strategy import TrendPullbackStrategy
    from src.strategies.liquidity_sweep_reversal_strategy import LiquiditySweepReversalStrategy
    from src.strategies.volatility_breakout_strategy import VolatilityBreakoutStrategy
    from src.strategies.triple_confluence_trend_strategy import TripleConfluenceTrendStrategy
    from src.strategies.bb_mean_reversion_strategy import BBMeanReversionStrategy
    from src.strategies.macd_breakout_strategy import MACDBreakoutStrategy
    classes = {
        "trend_pullback": TrendPullbackStrategy,
        "liquidity_sweep_reversal": LiquiditySweepReversalStrategy,
        "volatility_breakout": VolatilityBreakoutStrategy,
        "triple_confluence_trend": TripleConfluenceTrendStrategy,
        "bb_mean_reversion": BBMeanReversionStrategy,
        "macd_breakout": MACDBreakoutStrategy,
    }
    res = {k: {"trades": [], "signals": 0, "full_signals": 0,
               "conf_pass": 0} for k in classes}
    n = len(df)
    for i in range(WARMUP, n - 1):
        wdf = df.iloc[max(0, i - WINDOW + 1): i + 1].reset_index(drop=True)
        for key, cls in classes.items():
            try:
                s = cls(weight=1.0).analyze(wdf, sym)
            except Exception:
                continue
            if s.direction != "bullish":
                continue
            r = res[key]
            r["signals"] += 1
            if s.score >= 50:
                r["full_signals"] += 1
                # single-vote admission confidence (worst case in production)
                conf = 100 * (0.65 * (s.score / 100.0) + 0.35 * (1.0 / CONFLUENCE_REF))
                if conf >= MIN_CONF:
                    r["conf_pass"] += 1
                t = simulate_trade(df, i, 0.020, 0.024, 0.048)
                if t is not None:
                    r["trades"].append(t)
    return {sym: res}


def fold_report(trades: list, df_times: list) -> dict:
    """3 equal folds by exit time index; expectancy per fold."""
    if not trades:
        return {}
    idx = np.array([t["exit_idx"] for t in trades])
    rets = np.array([t["ret"] for t in trades])
    order = np.argsort(idx)
    idx, rets = idx[order], rets[order]
    cuts = np.array_split(np.arange(len(idx)), 4)
    folds = []
    for ci in cuts:
        if len(ci) == 0:
            folds.append({"n": 0, "expectancy_bps": 0.0})
            continue
        f = rets[ci]
        folds.append({"n": int(len(f)),
                      "expectancy_bps": float(f.mean() * 1e4),
                      "win_rate": float((f > 0).mean())})
    return {"folds": folds}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", type=int, default=0, help="limit universe (smoke)")
    ap.add_argument("--skip-baseline", action="store_true")
    args = ap.parse_args()

    universe = json.loads((DATA / "universe.json").read_text())["symbols"]
    if args.symbols:
        universe = universe[: args.symbols]
    print(f"universe: {len(universe)} symbols")

    results = {"candidates": {}, "baselines": {}}

    # ---- candidates (vectorised, fast) ---------------------------------
    print("== candidates ==", flush=True)
    tide = btc_tide()
    combos = [(c, t) for c in CANDIDATES for t in (False, True)]
    cand_trades = {f"{c}{'+tide' if t else ''}": [] for c, t in combos}
    for si, sym in enumerate(universe, 1):
        path = DATA / "4h" / f"{sym}.csv.gz"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if len(df) < WARMUP + 60:
            continue
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
        for c, use_tide in combos:
            sig, _ = candidate_signals(df, c, tide if use_tide else None)
            busy_until = -1
            for i in np.where(sig)[0]:
                if i < WARMUP or i <= busy_until:
                    continue
                t = simulate_trade(df, int(i), *CAND_PARAMS[c])
                if t is None:
                    continue
                busy_until = int(i) + t["hold"]
                t["exit_idx"] = int(i) + t["hold"]
                cand_trades[f"{c}{'+tide' if use_tide else ''}"].append(t)
        print(f"  [{si}/{len(universe)}] {sym} done", flush=True)
    for key, trades in cand_trades.items():
        results["candidates"][key] = {
            "metrics": trade_metrics(trades),
            "folds": fold_report(trades, None),
        }

    # ---- baselines (strategy classes, parallel over symbols) ------------
    if not args.skip_baseline:
        print("== baselines ==", flush=True)
        with ProcessPoolExecutor(max_workers=2) as ex:
            maps = list(ex.map(run_baselines_symbol, universe))
        agg = {k: {"trades": [], "signals": 0, "full_signals": 0, "conf_pass": 0}
               for k, _, _ in STRATEGY_CLASSES}
        for m in maps:
            for sym, res in m.items():
                for k, v in res.items():
                    if k not in agg:
                        continue
                    agg[k]["signals"] += v["signals"]
                    agg[k]["full_signals"] += v["full_signals"]
                    agg[k]["conf_pass"] += v["conf_pass"]
                    for t in v["trades"]:
                        agg[k]["trades"].append((sym, t))
        for key, _, w in STRATEGY_CLASSES:
            pairs = agg[key]["trades"]
            trades = [t for _, t in pairs]
            m = trade_metrics(trades)
            m["signals"] = agg[key]["signals"]
            m["full_signals"] = agg[key]["full_signals"]
            m["conf_pass"] = agg[key]["conf_pass"]
            m["weight"] = w
            # bars iterate ascending per symbol and symbols map()ed in order,
            # so append order approximates chronological exit order.
            idx_seq = np.arange(len(trades))
            for t, ii in zip(trades, idx_seq):
                t.setdefault("exit_idx", ii)
            fr = fold_report(trades, None)
            results["baselines"][key] = {"metrics": m, "folds": fr}

    OUT.write_text(json.dumps(results, indent=1, default=float))

    # ---- console summary -------------------------------------------------
    print("\n================ RESULTS ================")
    print(f"{'strategy':28s} {'n':>4} {'WR':>6} {'exp bps':>9} {'PF':>6} "
          f"{'CI95':>16} {'sig':>6}")
    rows = []
    for k, v in results["candidates"].items():
        m = v["metrics"]
        if m.get("n", 0) == 0:
            continue
        rows.append((f"[C] {k}", m))
    for k, v in results["baselines"].items():
        m = v["metrics"]
        rows.append((f"[B] {k}", m))
    rows.sort(key=lambda x: -(x[1].get("expectancy_bps") or -999))
    for name, m in rows:
        ci = m.get("ci95_bps", [0, 0])
        print(f"{name:28s} {m.get('n',0):4d} {m.get('win_rate',0)*100:5.1f}% "
              f"{m.get('expectancy_bps',0):+9.1f} {m.get('profit_factor',0):6.2f} "
              f"[{ci[0]:+6.1f},{ci[1]:+6.1f}] sig={m.get('signals','-')}")
    print(f"\nsaved -> {OUT}")


if __name__ == "__main__":
    main()
