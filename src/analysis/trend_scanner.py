"""Trend Scanner (v5.28) - evidence-based strategy replacement channel.

Research-driven replacement for the starved composite stack: 40 top-volume
USDT pairs x ~500 days of real 4h klines, honest costs (0.24% round trip),
the PRODUCTION exit lifecycle (TP1 50% partial + BE buffer + profit ladder
+ v5.27 fee-survival rung + time stops), 4 time-folds and bootstrap CIs
selected two long-only entry families with positive expectancy AFTER fees:

  1. donchian_break (Turtle/Donchian lineage):
       close crosses ABOVE the prior 55-bar high with volume >= 1.2x avg.
       n=1300, WR 63.7%, +28.3 bps/trade, PF 1.98, all 4 folds positive.

  2. st_flip (time-series momentum lineage, Liu & Tsyvinski 2021/2022):
       the PRODUCTION SuperTrend(10,3) flips up while close > EMA200.
       n=500, WR 51.6%, +14.7 bps/trade, PF 1.50, all 4 folds positive.

Rejected by the same harness: RSI(2) dip-buying (negative expectancy) and
BB-squeeze breakout (CI crosses zero) - see scripts/research/backtest.py.

Channel contract (same boost pattern as bottom/momentum channels):
  analyzer cycle -> trend_scanner.scan(liquid head, 4h WS-cached candles)
                -> vectorised signal checks on CLOSED bars
                -> candidates >= TREND_MIN_SCORE -> build_trend_rec
                -> session gate + per-cycle cap + re-rank
                -> risk manager channel caps (_trend_channel_ok)

Cost discipline: 4h candles are served from the WS cache (TIMEFRAMES[0])
so the channel adds ZERO REST weight - unlike the 1m momentum scanner.
"""
import time
from pathlib import Path
from typing import Dict, List, Optional

from config.settings import settings
from src.core.data_fetcher import data_fetcher
from src.indicators.supertrend import supertrend
from src.indicators.technical import atr
from src.utils.helpers import now_utc, save_json, to_json_safe
from src.utils.logger import log

TREND_CANDIDATES_FILE = Path("data/trend_candidates.json")

# Per-signal-source freshness: never re-fire the same episode at a symbol.
_COOLDOWN_MIN = settings.TREND_SYMBOL_COOLDOWN_MIN


class TrendScanner:
    """Scans the primary 4h timeframe for Donchian-breakout and
    SuperTrend-flip continuation entries (long-only, closed bars only)."""

    def __init__(self):
        # {(symbol, source): ts of last fired signal}
        self._last_fired: Dict[tuple, float] = {}
        log.info(
            "[cyan]TrendScanner[/] initialized (v5.28 evidence channel: "
            f"Donchian {settings.TREND_DONCHIAN_PERIOD} + vol "
            f"{settings.TREND_VOL_MULT:g}x, SuperTrend "
            f"{settings.TREND_ST_PERIOD}/{settings.TREND_ST_MULT:g} > EMA"
            f"{settings.TREND_EMA_PERIOD}, geometry SL {settings.TREND_SL_ATR:g}/"
            f"TP1 {settings.TREND_TP1_ATR:g}/TP2 {settings.TREND_TP2_ATR:g} ATR "
            f"on {settings.TIMEFRAMES[0]}, long-only)")

    # ------------------------------------------------------------------
    def _on_cooldown(self, symbol: str, source: str) -> bool:
        ts = self._last_fired.get((symbol, source))
        if ts is None:
            return False
        return (time.time() - ts) < _COOLDOWN_MIN * 60.0

    def _gate_market(self, df, atr_val: float) -> Optional[str]:
        """Fee-food / untradeable-market gates (4h scale)."""
        price = float(df["close"].iloc[-1])
        if price <= 0:
            return "bad price"
        atr_pct = atr_val / price * 100.0
        if atr_pct < settings.TREND_MIN_ATR_PCT:
            return (f"dead market (4h ATR {atr_pct:.3f}% < "
                    f"{settings.TREND_MIN_ATR_PCT:.2f}%)")
        if atr_pct > settings.TREND_MAX_ATR_PCT:
            return (f"volatility extreme (4h ATR {atr_pct:.2f}% > "
                    f"{settings.TREND_MAX_ATR_PCT:.2f}%)")
        return None

    # ------------------------------------------------------------------
    def scan(self, symbols: Optional[List[str]] = None,
             max_candidates: int = 6) -> List[Dict]:
        """Scan the liquid head on the primary timeframe; return candidates."""
        if not settings.TREND_BOOST_ENABLED:
            return []
        if symbols is None:
            from src.analysis.analyzer import analyzer
            symbols = analyzer.symbols
        shortlist = [s for s in (symbols or [])
                     if not self._on_cooldown(s, "donchian")
                     and not self._on_cooldown(s, "st_flip")]
        shortlist = shortlist[:max(1, settings.TREND_TOP_N)]
        if not shortlist:
            return []

        log.info(
            f"[cyan]Trend scanning[/] {len(shortlist)} symbols on "
            f"{settings.TIMEFRAMES[0]} (donchian+st_flip)...")
        start = time.time()

        candidates: List[Dict] = []
        for symbol in shortlist:
            try:
                found = self._scan_one(symbol)
            except Exception as e:
                log.debug(f"Trend scan error {symbol}: {e}")
                continue
            for c in found:
                candidates.append(c)

        candidates.sort(key=lambda c: c.get("score", 0), reverse=True)
        top = candidates[:max_candidates]

        save_json(to_json_safe({
            "timestamp": now_utc().isoformat(),
            "timeframe": settings.TIMEFRAMES[0],
            "symbols_scanned": len(shortlist),
            "candidates_found": len(candidates),
            "top_candidates": top,
            "scan_time_seconds": round(time.time() - start, 2),
        }), TREND_CANDIDATES_FILE)

        if top:
            log.info(
                f"[green]Trend scan[/] - {len(top)} trend candidate(s) "
                f"in {time.time()-start:.1f}s")
        return top

    # ------------------------------------------------------------------
    def _scan_one(self, symbol: str) -> List[Dict]:
        tf = settings.TIMEFRAMES[0]
        df = data_fetcher.get_candles(symbol, tf,
                                      limit=settings.TREND_KLINES_LIMIT)
        if df is None or len(df) < settings.TREND_EMA_PERIOD + 2:
            return []
        # Signals are computed on CLOSED bars only: drop the still-forming
        # candle exactly like the double-indicator channel does (frames
        # without a close_time column - tests, WS cache - are assumed
        # closed already).
        from src.strategies.double_indicator_strategy import last_closed_bars
        df = last_closed_bars(df)
        if df is None or len(df) < settings.TREND_EMA_PERIOD + 1:
            return []

        c = df["close"]
        vol = df["volume"]
        atr_series = atr(df["high"], df["low"], c, 14)
        atr_val = float(atr_series.iloc[-1])
        if atr_val <= 0:
            return []
        skip = self._gate_market(df, atr_val)
        if skip:
            return []
        price = float(c.iloc[-1])

        ema200 = c.ewm(span=settings.TREND_EMA_PERIOD, adjust=False,
                       min_periods=settings.TREND_EMA_PERIOD).mean()
        above_ema = bool(price > float(ema200.iloc[-1]))
        vol_ratio = float(vol.iloc[-1] / max(vol.rolling(20).mean().iloc[-1],
                                             1e-12))
        out: List[Dict] = []

        # --- 1) Donchian channel breakout (cross semantics, stateless) ---
        if not self._on_cooldown(symbol, "donchian"):
            # channel = PRIOR 55-bar high (shift(1) excludes the current
            # bar, else close > rolling-max-of-its-own-bar is impossible)
            hh = df["high"].rolling(
                settings.TREND_DONCHIAN_PERIOD).max().shift(1)
            cross = (c > hh) & (c.shift(1) <= hh.shift(1)) \
                & (vol > settings.TREND_VOL_MULT * vol.rolling(20).mean())
            if bool(cross.iloc[-1]):
                score = self._score_breakout(vol_ratio, above_ema, atr_val,
                                             price, float(hh.iloc[-1]))
                if score >= settings.TREND_MIN_SCORE:
                    self._last_fired[(symbol, "donchian")] = time.time()
                    out.append(self._candidate(
                        symbol, "donchian", tf, score, price, atr_val,
                        vol_ratio, above_ema, reasons=[
                            f"close {price:.6g} crossed the "
                            f"{settings.TREND_DONCHIAN_PERIOD}-bar high "
                            f"{float(hh.iloc[-1]):.6g}",
                            f"volume {vol_ratio:.2f}x avg (>= "
                            f"{settings.TREND_VOL_MULT:g}x required)",
                            f"close {'above' if above_ema else 'below'} "
                            f"EMA{settings.TREND_EMA_PERIOD}",
                        ]))

        # --- 2) SuperTrend flip continuation (production indicator) ------
        if not self._on_cooldown(symbol, "st_flip"):
            _, trend = supertrend(df["high"], df["low"], c,
                                  settings.TREND_ST_PERIOD,
                                  settings.TREND_ST_MULT)
            flip = (trend == 1) & (trend.shift(1) == -1)
            if bool(flip.iloc[-1]) and above_ema:
                score = self._score_flip(vol_ratio, above_ema, atr_val, price,
                                         float(ema200.iloc[-1]))
                if score >= settings.TREND_MIN_SCORE:
                    self._last_fired[(symbol, "st_flip")] = time.time()
                    out.append(self._candidate(
                        symbol, "st_flip", tf, score, price, atr_val,
                        vol_ratio, above_ema, reasons=[
                            f"SuperTrend({settings.TREND_ST_PERIOD},"
                            f"{settings.TREND_ST_MULT:g}) flipped bullish",
                            f"close {price:.6g} above EMA"
                            f"{settings.TREND_EMA_PERIOD} "
                            f"{float(ema200.iloc[-1]):.6g}",
                            f"volume {vol_ratio:.2f}x avg",
                        ]))
        return out

    # ------------------------------------------------------------------
    def _score_breakout(self, vol_ratio: float, above_ema: bool,
                        atr_val: float, price: float,
                        channel_high: float) -> int:
        """62..85: breakout quality on volume, regime and distance."""
        score = 62
        if vol_ratio >= 2.0:
            score += 10
        elif vol_ratio >= 1.5:
            score += 6
        if above_ema:
            score += 8
        # A fresh, close breakout (just above the channel) beats a 3-ATR
        # moonshot that is already due for a retest.
        dist_atr = (price - channel_high) / max(atr_val, 1e-12)
        if dist_atr <= 0.35:
            score += 6
        elif dist_atr > 1.0:
            score -= 6
        return int(max(0, min(85, score)))

    def _score_flip(self, vol_ratio: float, above_ema: bool,
                    atr_val: float, price: float, ema_val: float) -> int:
        """62..85: flip quality on regime distance and tape."""
        score = 62
        if above_ema:
            score += 8
        # Distance above EMA200 in ATR terms: a flip just above the mean
        # regime line is early and cheap; far above is a late chase.
        dist_atr = (price - ema_val) / max(atr_val, 1e-12)
        if dist_atr <= 1.5:
            score += 6
        elif dist_atr > 4.0:
            score -= 6
        if vol_ratio >= 1.5:
            score += 6
        elif vol_ratio < 0.8:
            score -= 4
        return int(max(0, min(85, score)))

    def _candidate(self, symbol: str, source: str, tf: str, score: int,
                   price: float, atr_val: float, vol_ratio: float,
                   above_ema: bool, reasons: List[str]) -> Dict:
        """Candidate dict with the research-validated geometry."""
        sl = price - settings.TREND_SL_ATR * atr_val
        tp1 = price + settings.TREND_TP1_ATR * atr_val
        tp2 = price + settings.TREND_TP2_ATR * atr_val
        rr = abs(tp2 - price) / max(abs(price - sl), 1e-9)
        return {
            "symbol": symbol,
            "source": f"trend_{source}",
            "score": int(score),
            "direction": "bullish",
            "timeframe": tf,
            "current_price": price,
            "stop_loss": float(sl),
            "take_profit": float(tp1),
            "take_profit_2": float(tp2),
            "risk_reward_ratio": float(rr),
            "atr": atr_val,
            "atr_pct": float(atr_val / price * 100) if price > 0 else 0.0,
            "volume_ratio": vol_ratio,
            "above_ema200": above_ema,
            "signals": reasons,
            "analyzed_at": now_utc().isoformat(),
        }


# Singleton
trend_scanner = TrendScanner()
