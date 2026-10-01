"""Momentum Scanner (v5.22) - the user's double-indicator strategy channel.

Scans the most liquid head of the universe on DOUBLE_IND_TIMEFRAME (1m)
candles and hands the strategy's long-only signals to the recommendation
pipeline (same boost pattern as the bottom scanner):

  analyzer cycle -> momentum_scanner.scan(top-N liquid symbols)
               -> DoubleIndicatorStrategy.analyze(1m closed bars)
               -> candidates >= 100% checklist -> build_double_ind_rec
               -> filtered recommendations -> paper/live positions

Cost discipline (the bot lives on a shared Render IP with a 418 history):
  * ban guard: zero REST calls while a 429/418 cooldown is active;
  * TOP-N shortlist (15) x weight-2 klines calls (limit 90) = 30 REST
    weight per cycle - the bottom scanner spends 384;
  * per-symbol cooldown stops the same 3-candle burst from re-firing
    cycle after cycle while the same momentum episode is still running.
"""
import time
from pathlib import Path
from typing import Dict, List, Optional

from config.settings import settings
from src.core.data_fetcher import data_fetcher
from src.indicators.technical import atr
from src.strategies.double_indicator_strategy import (
    DoubleIndicatorStrategy, last_closed_bars)
from src.utils.helpers import now_utc, save_json, to_json_safe
from src.utils.logger import log

MOMENTUM_CANDIDATES_FILE = Path("data/momentum_candidates.json")


class MomentumScanner:
    """Scans 1m candles for double-indicator (BB 11/3 + SuperTrend 2/2)
    long-only momentum entries."""

    def __init__(self):
        self._strategy = DoubleIndicatorStrategy()
        # {symbol: ts of last fired signal} - burst dedup across cycles
        self._last_fired: Dict[str, float] = {}
        log.info(
            "[cyan]MomentumScanner[/] initialized (double_indicator: "
            f"BB {settings.DOUBLE_IND_BB_PERIOD}/{settings.DOUBLE_IND_BB_DEV:g} "
            f"+ SuperTrend {settings.DOUBLE_IND_ST_PERIOD}/"
            f"{settings.DOUBLE_IND_ST_MULT:g} on "
            f"{settings.DOUBLE_IND_TIMEFRAME}, long-only)")

    # ------------------------------------------------------------------
    def _symbol_on_cooldown(self, symbol: str) -> bool:
        ts = self._last_fired.get(symbol)
        if ts is None:
            return False
        return (time.time() - ts) < settings.DOUBLE_IND_SYMBOL_COOLDOWN_MIN * 60.0

    def _gate_candidate(self, df, symbol: str) -> Optional[str]:
        """Semi-stable / fee-food gates (1m scale). Returns skip reason."""
        price = float(df["close"].iloc[-1])
        if price <= 0:
            return "bad price"
        atr_val = float(atr(df["high"], df["low"], df["close"], 14).iloc[-1])
        if atr_val <= 0:
            return "no ATR"
        atr_pct = atr_val / price * 100.0
        if atr_pct < settings.DOUBLE_IND_MIN_ATR_PCT:
            return (f"semi-stable/dead coin (1m ATR {atr_pct:.3f}% < "
                    f"{settings.DOUBLE_IND_MIN_ATR_PCT:.2f}% floor)")
        lb = min(60, len(df))
        tail = df.tail(lb)
        rng_pct = ((float(tail["high"].max()) - float(tail["low"].min()))
                   / price * 100.0)
        if rng_pct < settings.DOUBLE_IND_MIN_RANGE_PCT:
            return (f"flat/pinned range ({rng_pct:.2f}% < "
                    f"{settings.DOUBLE_IND_MIN_RANGE_PCT:.2f}% over "
                    f"{lb} bars)")
        return None

    # ------------------------------------------------------------------
    def scan(self, symbols: Optional[List[str]] = None,
             max_candidates: int = 5) -> List[Dict]:
        """Scan the liquid shortlist; return double-indicator candidates."""
        if not settings.DOUBLE_IND_ENABLED:
            return []
        # v5.12 pattern: a hard 429/418 ban makes every REST klines call
        # here doomed - bail out cleanly, the next cycle retries.
        try:
            from src.core.rate_limiter import rate_limiter
            if rate_limiter.cooldown_remaining() > 130.0:
                log.warning(
                    f"[yellow]Momentum scanner skipped[/] - Binance rate "
                    f"ban active ({rate_limiter.cooldown_remaining():.0f}s left)"
                )
                return []
        except Exception:
            pass
        if symbols is None:
            from src.analysis.analyzer import analyzer
            symbols = analyzer.symbols
        # Most liquid head only (the dynamic universe is volume-sorted)
        shortlist = [s for s in (symbols or [])
                     if not self._symbol_on_cooldown(s)]
        shortlist = shortlist[:max(1, settings.DOUBLE_IND_TOP_N)]
        if not shortlist:
            return []

        log.info(
            f"[cyan]Momentum scanning[/] {len(shortlist)} symbols on "
            f"{settings.DOUBLE_IND_TIMEFRAME} (double_indicator)...")
        start = time.time()

        candidates: List[Dict] = []
        for symbol in shortlist:
            try:
                c = self._scan_one(symbol)
            except Exception as e:
                log.debug(f"Momentum scan error {symbol}: {e}")
                continue
            if c and not c.get("skip"):
                candidates.append(c)

        candidates.sort(key=lambda c: c.get("score", 0), reverse=True)
        top = candidates[:max_candidates]

        save_json(to_json_safe({
            "timestamp": now_utc().isoformat(),
            "timeframe": settings.DOUBLE_IND_TIMEFRAME,
            "symbols_scanned": len(shortlist),
            "candidates_found": len(candidates),
            "top_candidates": top,
            "scan_time_seconds": round(time.time() - start, 2),
        }), MOMENTUM_CANDIDATES_FILE)

        if top:
            log.info(
                f"[green]Momentum scan[/] - {len(top)} double-indicator "
                f"candidate(s) in {time.time()-start:.1f}s")
        return top

    def _scan_one(self, symbol: str) -> Optional[Dict]:
        df = data_fetcher.get_candles(
            symbol, settings.DOUBLE_IND_TIMEFRAME,
            limit=settings.DOUBLE_IND_KLINES_LIMIT)
        df = last_closed_bars(df)
        if df is None or len(df) < 30:
            return {"symbol": symbol, "skip": True, "reason": "insufficient data"}

        skip = self._gate_candidate(df, symbol)
        if skip:
            return {"symbol": symbol, "skip": True, "reason": skip}

        sig = self._strategy.analyze(df, symbol)
        if sig.direction != "bullish":
            return {"symbol": symbol, "skip": True,
                    "reason": "; ".join(sig.reasons)[:180]}

        d = sig.details or {}
        price = float(df["close"].iloc[-1])
        sl = float(d["stop_loss"])
        tp1 = float(d["take_profit"])
        tp2 = float(d["take_profit_2"])
        rr = abs(tp2 - price) / max(abs(price - sl), 1e-9)
        atr_val = float(d.get("atr") or 0.0)

        self._last_fired[symbol] = time.time()
        return {
            "symbol": symbol,
            "score": float(d.get("score", sig.score)),
            "direction": "bullish",
            "timeframe": settings.DOUBLE_IND_TIMEFRAME,
            "current_price": price,
            "stop_loss": sl,
            "take_profit": tp1,
            "take_profit_2": tp2,
            "risk_reward_ratio": float(rr),
            "atr": atr_val,
            "atr_pct": float(atr_val / price * 100) if price > 0 else 0.0,
            "pct_b_last": d.get("pct_b_last"),
            "pct_b_avg": d.get("pct_b_avg"),
            "volume_ratio": d.get("volume_ratio"),
            "st_line": d.get("st_line"),
            "signals": sig.reasons,
            "green_run": d.get("green_run"),
            "analyzed_at": now_utc().isoformat(),
        }


# Singleton
momentum_scanner = MomentumScanner()
