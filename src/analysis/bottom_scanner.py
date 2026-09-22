"""
Bottom Scanner - Searches for coins that are near their recent lows
and showing potential bounce signals.

Strategy:
  1. Filter all USDT pairs to those within 5% of their 30-day low
  2. Score each based on bounce signals:
     - RSI oversold (< 35)
     - Volume climax (capitulation)
     - Bullish candlestick pattern detected
     - Wyckoff Spring present
     - Smart Money Divergence
     - Fibonacci golden zone
  3. Sort by score, return top 20 bounce candidates
"""
import time
from pathlib import Path
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
from config.settings import settings
from src.core.data_fetcher import data_fetcher
from src.indicators.technical import rsi, bollinger_bands, atr
from src.indicators.proprietary import (
    is_near_bottom, volume_climax, wyckoff_spring,
    smart_money_divergence, fibonacci_confluence
)
from src.indicators.patterns import detect_all_patterns
from src.utils.logger import log
from src.utils.helpers import to_json_safe, save_json, now_utc

BOTTOM_CANDIDATES_FILE = Path("data/bottom_candidates.json")


def score_bottom_candidate(symbol: str, df: pd.DataFrame) -> Dict:
    """Score a coin as a bottom-reversal candidate."""
    if len(df) < 60:
        return {"symbol": symbol, "skip": True}

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    current_price = float(close.iloc[-1])

    # Check if near bottom
    bottom_check = is_near_bottom(df, lookback=50, threshold_pct=0.05)
    if not bottom_check["near_bottom"]:
        return {
            "symbol": symbol, "skip": True,
            "reason": f"Not near bottom (distance {bottom_check['distance_from_low_pct']:.1f}%)"
        }

    score = 30  # Base score for being near bottom
    signals = []

    # === RSI oversold ===
    rsi_val = float(rsi(close, 14).iloc[-1])
    if rsi_val < 30:
        score += 25
        signals.append(f"RSI deeply oversold ({rsi_val:.1f})")
    elif rsi_val < 40:
        score += 10
        signals.append(f"RSI oversold ({rsi_val:.1f})")

    # === Volume climax (capitulation) ===
    climax = volume_climax(df, lookback=50, spike_threshold=2.0, drop_threshold=-1.5)
    if climax.get("signal") == "bullish":
        score += climax.get("strength", 0.5) * 40
        signals.append(climax.get("reason", ""))

    # === Wyckoff Spring ===
    spring = wyckoff_spring(df, support_lookback=50)
    if spring.get("signal") == "bullish":
        score += spring.get("strength", 0.5) * 45
        signals.append(spring.get("reason", ""))

    # === Smart Money Divergence ===
    smd = smart_money_divergence(close, volume, lookback=50)
    if smd.get("signal") == "bullish":
        score += smd.get("strength", 0.5) * 35
        signals.append(smd.get("reason", ""))

    # === Fibonacci golden zone ===
    fib = fibonacci_confluence(df, lookback=100)
    if fib.get("signal") == "bullish":
        score += fib.get("strength", 0.5) * 30
        signals.append(fib.get("reason", ""))

    # === Bullish candlestick patterns ===
    patterns = detect_all_patterns(df)
    bullish_patterns = [p for p in patterns if p.get("direction") == "bullish"]
    if bullish_patterns:
        pattern_score = sum(p["strength"] for p in bullish_patterns) * 30
        score += min(30, pattern_score)
        for p in bullish_patterns[:3]:
            signals.append(f"{p['pattern']} ({p['strength']:.1f})")

    # === Bollinger %B (below lower band) ===
    bb = bollinger_bands(close, 20, 2)
    pct_b = float(bb["percent_b"].iloc[-1])
    if pct_b < 0.05:
        score += 15
        signals.append(f"Below lower BB (Percent B = {pct_b:.2f})")

    # === Confirmation: last candle closed bullish (no falling knives) ===
    last_candle_bullish = bool(df["close"].iloc[-1] > df["open"].iloc[-1])

    # Compute suggested SL/TP using ATR
    atr_val = float(atr(high, low, close, 14).iloc[-1])
    sl_distance = atr_val * 1.2  # tighter SL for bottom fishing
    tp_distance = atr_val * 2.5
    stop_loss = current_price - sl_distance
    take_profit = current_price + tp_distance
    rr_ratio = abs(take_profit - current_price) / max(abs(current_price - stop_loss), 0.0001)

    return {
        "symbol": symbol,
        "score": float(score),
        "current_price": current_price,
        "recent_low": bottom_check["recent_low"],
        "recent_high": bottom_check["recent_high"],
        "distance_from_low_pct": bottom_check["distance_from_low_pct"],
        "position_in_range_pct": bottom_check["position_in_range_pct"],
        "rsi": rsi_val,
        "atr_pct": float(atr_val / current_price * 100),
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "risk_reward_ratio": float(rr_ratio),
        "signals": signals,
        "bb_percent_b": pct_b,
        "last_candle_bullish": last_candle_bullish,
        "patterns_detected": [p["pattern"] for p in patterns],
        "analyzed_at": now_utc().isoformat(),
    }


class BottomScanner:
    """Scans the market for coins near their bottoms with bounce signals."""

    def __init__(self):
        log.info("[cyan]BottomScanner[/] initialized")

    def scan(self, symbols: List[str] = None, max_candidates: int = 20,
              limit: int = 200) -> List[Dict]:
        """
        Scan all symbols for bottom-reversal candidates.
        Returns top 20 candidates sorted by score.
        """
        if symbols is None:
            from src.analysis.analyzer import analyzer
            analyzer.refresh_symbols()
            symbols = analyzer.symbols

        log.info(f"[cyan]Bottom scanning[/] {len(symbols)} symbols...")
        start = time.time()

        results = []
        with ThreadPoolExecutor(max_workers=settings.MAX_WORKERS) as ex:
            futures = {ex.submit(self._scan_one, s, limit): s for s in symbols}
            for fut in as_completed(futures):
                try:
                    r = fut.result()
                    if r and not r.get("skip"):
                        results.append(r)
                except Exception as e:
                    log.debug(f"Scan error: {e}")

        results.sort(key=lambda r: r.get("score", 0), reverse=True)
        top = results[:max_candidates]

        log.info(
            f"[green]Bottom scan complete[/] - {len(results)} candidates found, "
            f"top {len(top)} returned in {time.time()-start:.1f}s"
        )

        save_data = {
            "timestamp": now_utc().isoformat(),
            "symbols_scanned": len(symbols),
            "candidates_found": len(results),
            "top_candidates": top,
            "scan_time_seconds": round(time.time() - start, 2),
        }
        save_json(to_json_safe(save_data), BOTTOM_CANDIDATES_FILE)
        return top

    def _scan_one(self, symbol: str, limit: int) -> Dict:
        """Scan one symbol."""
        try:
            df = data_fetcher.get_candles(symbol, settings.TIMEFRAMES[0], limit=limit)
            if df is None or df.empty or len(df) < 60:
                return {"symbol": symbol, "skip": True}
            return score_bottom_candidate(symbol, df)
        except Exception as e:
            return {"symbol": symbol, "skip": True, "reason": str(e)}


# Singleton
bottom_scanner = BottomScanner()
