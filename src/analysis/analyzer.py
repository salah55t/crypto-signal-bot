"""
Main Analyzer - orchestrates data fetching + multi-strategy analysis
across all configured symbols.
"""
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from config.settings import settings
from src.core.data_fetcher import data_fetcher
from src.analysis.scorer import scorer
from src.db.database import db
from src.utils.logger import log
from src.utils.helpers import save_json, now_utc, to_json_safe

RECOMMENDATIONS_FILE = Path("data/recommendations.json")


class MarketAnalyzer:
    """Top-level orchestrator: fetch -> analyze -> score -> filter."""

    def __init__(self):
        self.excluded = settings.load_excluded()
        self._symbols_ts = None  # ts of last dynamic symbol-list refresh
        if settings.USE_ALL_USDT_PAIRS:
            # Fetch all USDT pairs dynamically from Binance
            self.symbols = self._fetch_all_usdt_pairs()
            self._symbols_ts = now_utc()
            log.info(
                f"[cyan]MarketAnalyzer[/] DYNAMIC MODE - monitoring "
                f"{len(self.symbols)} USDT pairs from Binance "
                f"(filtered by ${settings.MIN_VOLUME_USDT:,.0f} 24h volume, "
                f"cap: {settings.MAX_SYMBOLS})"
            )
        else:
            # Use the static list from config/coins.yaml
            self.symbols = settings.load_coins()
            self.symbols = [s for s in self.symbols if s not in self.excluded]
            log.info(
                f"[cyan]MarketAnalyzer[/] STATIC MODE - monitoring "
                f"{len(self.symbols)} symbols from config/coins.yaml"
            )

    def _fetch_all_usdt_pairs(self) -> list:
        """Fetch all USDT spot pairs from Binance, filtered by volume."""
        try:
            from src.core.binance_client import binance_client
            tickers = binance_client.get_all_tickers()
            # Single pass: USDT pairs with sufficient volume, not excluded,
            # sorted by 24h quote volume (descending), capped at MAX_SYMBOLS.
            sorted_pairs = sorted(
                (
                    t for t in tickers
                    if t.get("symbol", "").endswith("USDT")
                    and float(t.get("quoteVolume", 0)) >= settings.MIN_VOLUME_USDT
                    and t.get("symbol") not in self.excluded
                ),
                key=lambda t: float(t.get("quoteVolume", 0)),
                reverse=True,
            )
            return [t["symbol"] for t in sorted_pairs[:settings.MAX_SYMBOLS]]
        except Exception as e:
            log.error(f"Failed to fetch USDT pairs: {e}")
            # Fallback to static list
            return [s for s in settings.load_coins() if s not in self.excluded]

    def refresh_symbols(self, force: bool = False):
        """Re-fetch the symbol list (dynamic mode only).

        v5.3: cached for SYMBOL_REFRESH_MIN minutes. The old behaviour
        refetched /ticker/24hr (weight 80!) every cycle AND again from
        bottom_scanner.scan() - pure waste when symbols rarely churn.
        """
        if not settings.USE_ALL_USDT_PAIRS:
            return
        if not force and self._symbols_ts is not None:
            age = now_utc() - self._symbols_ts
            if age < timedelta(minutes=max(1, settings.SYMBOL_REFRESH_MIN)):
                return
        self.symbols = self._fetch_all_usdt_pairs()
        self._symbols_ts = now_utc()
        log.info(f"[cyan]Symbols refreshed[/] - {len(self.symbols)} pairs")
        # v5.3: keep the WS candle feed subscribed to the current universe
        try:
            from src.core.ws_feed import ws_feed
            ws_feed.update_universe(self.symbols)
        except Exception:
            pass

    def analyze_one(self, symbol: str) -> Dict:
        """Fetch data and run analysis for one symbol."""
        try:
            # Fetch multi-timeframe candles
            multi_tf = data_fetcher.get_multi_timeframe_candles(
                symbol, settings.TIMEFRAMES, limit=settings.CANDLE_LIMIT
            )
            if not multi_tf:
                return {"symbol": symbol, "skip": True, "reason": "No candle data"}

            # Primary timeframe = middle (usually 1h)
            primary_tf = settings.TIMEFRAMES[len(settings.TIMEFRAMES) // 2] \
                if len(settings.TIMEFRAMES) > 1 else settings.TIMEFRAMES[0]
            df = multi_tf.get(primary_tf)
            if df is None or len(df) < 60:
                return {"symbol": symbol, "skip": True, "reason": "Insufficient primary data"}

            # Fetch order book (only if not skipped for performance)
            order_book = None
            if not settings.SKIP_ORDER_BOOK:
                try:
                    order_book = data_fetcher.get_order_book(
                        symbol, limit=settings.ORDER_BOOK_DEPTH
                    )
                except Exception as e:
                    log.debug(f"Order book fetch failed for {symbol}: {e}")
                    order_book = None

            # Run analysis
            result = scorer.analyze_symbol(
                df, symbol,
                multi_tf_data=multi_tf,
                order_book=order_book,
            )
            result["timeframe"] = primary_tf
            result["analyzed_at"] = now_utc().isoformat()
            return result
        except Exception as e:
            log.error(f"Analysis failed for {symbol}: {e}")
            return {"symbol": symbol, "skip": True, "reason": f"Error: {e}"}

    def analyze_all(self, parallel: bool = True, max_workers: int = None) -> List[Dict]:
        """Analyze all configured symbols."""
        if max_workers is None:
            max_workers = settings.MAX_WORKERS
        log.info(
            f"[cyan]Starting market analysis[/] for {len(self.symbols)} symbols "
            f"(parallel={parallel}, workers={max_workers}, timeframes={settings.TIMEFRAMES})"
        )
        start = time.time()

        # Refresh symbols list before each cycle (when in dynamic mode)
        if settings.USE_ALL_USDT_PAIRS:
            self.refresh_symbols()

        results = []
        if parallel and len(self.symbols) > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(self.analyze_one, s): s for s in self.symbols}
                for fut in as_completed(futures):
                    try:
                        r = fut.result()
                        if not r.get("skip"):
                            results.append(r)
                    except Exception as e:
                        log.error(f"Future error: {e}")
        else:
            for s in self.symbols:
                r = self.analyze_one(s)
                if not r.get("skip"):
                    results.append(r)

        log.info(
            f"[green]Analysis complete[/] - {len(results)}/{len(self.symbols)} "
            f"symbols analyzed in {time.time()-start:.1f}s"
        )

        # v5.2 Market Map: classify followers to leaders (BTC/ETH/SOL/XRP) and
        # apply the leader's trend BEFORE filtering:
        #   bearish leader -> confidence penalty (admission too - may demote out)
        #   bullish leader -> rank-only boost (admission stays merit-based)
        try:
            from src.analysis.market_map import market_map
            results = market_map.apply_regime(results)
        except Exception as e:
            log.warning(f"[yellow]Market map step skipped: {e}[/]")

        # Filter: bullish only, sort by confidence
        filtered = scorer.filter_signals(
            results,
            min_confidence=settings.MIN_CONFIDENCE,
            min_expected_rise=settings.MIN_EXPECTED_RISE,
            direction="bullish",
        )
        log.info(
            f"[green]{len(filtered)} signals passed filter[/] "
            f"(confidence >= {settings.MIN_CONFIDENCE}%, expected rise >= {settings.MIN_EXPECTED_RISE}%)"
        )

        # === BOOST MECHANISM: Integrate Bottom Scanner candidates ===
        # The bottom scanner finds coins at recent lows with strong bounce signals.
        # SAFETY RULES (v2):
        #   - Only candidates with bounce score >= 60 (was 50 - too loose)
        #   - Confirmed only if the last candle closed bullish (no falling knives)
        #   - Max 2 boosted entries per cycle (they compete with real signals)
        #   - Confidence capped at 72 so genuine strategy signals rank first
        strong_bottoms = []
        try:
            from src.analysis.bottom_scanner import bottom_scanner
            log.info("[cyan]Running bottom scanner for boost mechanism...[/]")
            bottom_candidates = bottom_scanner.scan(max_candidates=20, limit=200)
            strong_bottoms = [
                c for c in bottom_candidates
                if c.get("score", 0) >= 60
                and c.get("last_candle_bullish", False)
            ]
            log.info(
                f"[green]{len(strong_bottoms)} confirmed bottom candidates[/] "
                f"(score >= 60 + bullish close)"
            )

            # === Log bottom candidates to database ===
            for c in strong_bottoms:
                try:
                    db.log_bottom_candidate(c)
                except Exception as e:
                    log.debug(f"Failed to log bottom candidate to DB: {e}")

            # Convert bottom candidates to recommendation format (if not already there)
            existing_symbols = {r.get("symbol") for r in filtered}
            boosted = 0
            for c in strong_bottoms:
                if boosted >= 2:
                    break
                if c["symbol"] in existing_symbols:
                    continue
                # Convert bottom candidate to recommendation format
                confidence = min(72.0, 45.0 + c["score"] / 3)
                rec = {
                    "symbol": c["symbol"],
                    "direction": "bullish",
                    "weighted_score": float(c["score"]),
                    "confidence": confidence,
                    "current_price": c["current_price"],
                    "expected_rise_pct": max(settings.MIN_EXPECTED_RISE,
                                              c["atr_pct"] * 1.8),  # ~1.8x ATR
                    "stop_loss": c["stop_loss"],
                    "take_profit": c["take_profit"],
                    "risk_reward_ratio": c["risk_reward_ratio"],
                    "atr": float(c.get("atr_pct", 0) * c["current_price"] / 100),
                    "atr_pct": c["atr_pct"],
                    "signals": [{
                        "strategy": "bottom_scanner_boost",
                        "direction": "bullish",
                        "score": c["score"],
                        "confidence": confidence / 100,
                        "reasons": c.get("signals", []),
                        "details": {
                            "bounce_score": c["score"],
                            "recent_low": c.get("recent_low"),
                            "distance_from_low_pct": c.get("distance_from_low_pct"),
                            "patterns_detected": c.get("patterns_detected", []),
                        }
                    }],
                    "timeframe": settings.TIMEFRAMES[0] if settings.TIMEFRAMES else "15m",
                    "analyzed_at": c.get("analyzed_at"),
                    "boosted_from_bottom": True,
                }
                # Boosted entries must pass the SAME confidence + R/R filters
                if (rec["confidence"] >= settings.MIN_CONFIDENCE
                        and rec["risk_reward_ratio"] >= settings.MIN_RR_RATIO):
                    filtered.append(rec)
                    boosted += 1
            if boosted:
                log.info(f"[green]{boosted} bottom candidates boosted into recommendations[/]")
        except Exception as e:
            log.error(f"Bottom scanner boost failed: {e}")

        # Limit
        top = filtered[:settings.MAX_RECOMMENDATIONS]

        # === Log to database ===
        try:
            duration = round(time.time() - start, 2)
            run_id = db.log_run(
                timestamp=now_utc().isoformat(),
                duration_seconds=duration,
                symbols_analyzed=len(self.symbols),
                signals_passed=len(filtered),
                recommendations_count=len(top),
                bottom_candidates_count=len(strong_bottoms) if 'strong_bottoms' in locals() else 0,
                mode=settings.RUN_MODE,
            )
            for rec in top:
                db.log_recommendation(run_id, rec)
            log.info(f"[green]Logged {len(top)} recommendations to database[/] (run_id={run_id})")
        except Exception as e:
            log.error(f"Failed to log to database: {e}")

        # Save full results
        save_data = {
            "timestamp": now_utc().isoformat(),
            "symbols_analyzed": len(self.symbols),
            "symbols_with_data": len(results),
            "signals_passed": len(filtered),
            "analysis_time_seconds": round(time.time() - start, 2),
            "top_recommendations": top,
            "all_results": results,
        }
        save_json(to_json_safe(save_data), RECOMMENDATIONS_FILE)
        log.info(f"Recommendations saved to {RECOMMENDATIONS_FILE}")

        return top


# Singleton
analyzer = MarketAnalyzer()
