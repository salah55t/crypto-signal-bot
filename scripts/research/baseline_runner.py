#!/usr/bin/env python3
"""Baseline runner with checkpointing (VM kills background jobs between
tool calls, so we run in foreground chunks and resume from partial JSON)."""
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, "/home/z/my-project/crypto-signal-bot")
sys.path.insert(0, "/home/z/my-project/crypto-signal-bot/scripts/research")

import backtest as bt  # noqa: E402

PARTIAL = bt.DATA / "baseline_partial.json"
BUDGET_S = 450  # stop gracefully before the tool timeout


def main():
    universe = json.loads((bt.DATA / "universe.json").read_text())["symbols"]
    done = {}
    if PARTIAL.exists():
        done = json.loads(PARTIAL.read_text())
    todo = [s for s in universe if s not in done]
    t0 = time.time()
    ex = ProcessPoolExecutor(max_workers=2)
    futures = {ex.submit(bt.run_baselines_symbol, s): s for s in todo}
    for fut, sym in futures.items():
        if time.time() - t0 > BUDGET_S:
            break
        try:
            res = fut.result(timeout=BUDGET_S + 60)
        except Exception as e:  # noqa: BLE001
            print(f"ERR {sym}: {e}", flush=True)
            continue
        done.update(res)
        PARTIAL.write_text(json.dumps(done))
        print(f"[{len(done)}/{len(universe)}] {sym} "
              f"({time.time()-t0:.0f}s)", flush=True)
    ex.shutdown(wait=False, cancel_futures=True)
    print(f"partial has {len(done)}/{len(universe)} symbols", flush=True)
    import os
    os._exit(0)  # bypass concurrent.futures' atexit join of running workers


if __name__ == "__main__":
    main()
