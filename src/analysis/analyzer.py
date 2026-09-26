"""
Main Analyzer - orchestrates data fetching + multi-strategy analysis
across all configured symbols.
"""
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from config.settings import settings
from src.core.data_fetcher import data_fetcher
from src.analysis.scorer import scorer
from src.db.database import db
from src.utils.logger import log
from src.utils.helpers import save_json, now_utc, to_json_safe

RECOMMENDATIONS_FILE = Path("data/recommendations.json")
# v5.15: last good dynamic USDT universe persisted to disk - a restart during
# a REST ban boots from this cache instead of poking a banned IP with a
# /ticker/24hr (weight 80) request.
SYMBOLS_CACHE_FILE = Path("data/symbols_cache.json")


def _ban_active(threshold: float = 130.0) -> bool:
    """v5.12: True while a hard Binance 429/418 cooldown is in force.

    Mirrors the retry_on_failure abort threshold (130s): below it calls
    still back off and retry; above it every REST call is doomed, so the
    whole burst should stop instead of grinding through the symbol list.
    """
    try:
        from src.core.rate_limiter import rate_limiter
        return rate_limiter.cooldown_remaining() > threshold
    except Exception:
        return False


def build_bottom_rec(c: Dict) -> Dict:
    """v5.6: convert a bottom-scanner candidate into a recommendation dict.

    Confidence uses the BOUNCE scale (40 + score/2, capped at
    BOTTOM_CONF_CAP) instead of the old 45 + score/3: with MIN_CONFIDENCE
    raised to 68, the old mapping silently required score >= 69 and killed
    the whole channel. A score-62 candidate now maps to 71% confidence.

    Harmony is derived from how many independent bounce layers agreed
    (RSI / climax / Wyckoff spring / SMD / fib / patterns / BB) - it feeds
    the composite ranking and makes the rec self-explanatory on the
    dashboard, while validate_recommendation still exempts boosted recs
    from the strategy-scale MIN_HARMONY gate.
    """
    score = float(c.get("score", 0))
    confidence = min(settings.BOTTOM_CONF_CAP, 40.0 + score / 2.0)
    n_layers = len(c.get("signals") or [])
    harmony = round(min(0.85, 0.35 + 0.10 * max(0, n_layers - 1)), 2)
    return {
        "symbol": c["symbol"],
        "direction": "bullish",
        "weighted_score": score,
        "confidence": confidence,
        "admission_confidence": confidence,
        "current_price": c["current_price"],
        "expected_rise_pct": max(settings.MIN_EXPECTED_RISE,
                                  c["atr_pct"] * 1.8),  # ~1.8x ATR
        "stop_loss": c["stop_loss"],
        "take_profit": c["take_profit"],
        "risk_reward_ratio": c["risk_reward_ratio"],
        "atr": float(c.get("atr_pct", 0) * c["current_price"] / 100),
        "atr_pct": c["atr_pct"],
        "harmony": harmony,
        "signals": [{
            "strategy": "bottom_scanner_boost",
            "direction": "bullish",
            "score": score,
            "confidence": confidence / 100,
            "reasons": c.get("signals", []),
            "details": {
                "bounce_score": score,
                "recent_low": c.get("recent_low"),
                "distance_from_low_pct": c.get("distance_from_low_pct"),
                "patterns_detected": c.get("patterns_detected", []),
            }
        }],
        "timeframe": settings.TIMEFRAMES[0] if settings.TIMEFRAMES else "15m",
        "analyzed_at": c.get("analyzed_at"),
        "boosted_from_bottom": True,
    }


class MarketAnalyzer:
    """Top-level orchestrator: fetch -> analyze -> score -> filter."""

    def __init__(self):
        self.excluded = settings.load_excluded()
        self._symbols_ts = None  # ts of last dynamic symbol-list refresh
        # v5.12: set when the analysis burst was aborted by a Binance rate
        # ban - cycle.py checks it to skip notifications and keep the last
        # good recommendations snapshot intact.
        self.last_run_aborted = False
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

    def _load_symbols_cache(self) -> Optional[List[str]]:
        """v5.15: last good dynamic USDT pair list from disk (ban-safe boot).

        data/symbols_cache.json = {"symbols": [...], "saved_at": iso, ...
        Accepted while younger than settings.SYMBOLS_CACHE_MAX_AGE_H so a
        restart during a REST ban keeps the FULL dynamic universe instead of
        silently shrinking to the static coins.yaml list.
        """
        try:
            from src.utils.helpers import load_json, now_utc
            payload = load_json(SYMBOLS_CACHE_FILE, default=None)
            if not isinstance(payload, dict):
                return None
            syms = payload.get("symbols")
            saved = payload.get("saved_at")
            if not isinstance(syms, list) or not syms:
                return None
            if saved:
                saved_dt = datetime.fromisoformat(str(saved))
                if saved_dt.tzinfo is None:  # tolerate naive timestamps
                    saved_dt = saved_dt.replace(tzinfo=timezone.utc)
                age_h = (now_utc() - saved_dt).total_seconds() / 3600.0
                if age_h > max(1.0, float(getattr(
                    settings, "SYMBOLS_CACHE_MAX_AGE_H", 168
                ))):
                    return None
            return [str(s) for s in syms if s]
        except Exception:
            return None

    def _save_symbols_cache(self, symbols: List[str]) -> None:
        """v5.15: persist the dynamic universe so restarts don't need REST."""
        try:
            save_json(
                {
                    "symbols": list(symbols),
                    "saved_at": now_utc().isoformat(),
                    "count": len(symbols),
                },
                SYMBOLS_CACHE_FILE,
            )
        except Exception as e:
            log.debug(f"Symbols cache write failed: {e}")

    def _fetch_all_usdt_pairs(self) -> list:
        """Fetch all USDT spot pairs from Binance, filtered by volume.

        v5.15: BAN-SAFE. Called from __init__ (import time!) and every
        SYMBOL_REFRESH_MIN - both fire straight into a still-active Binance
        418 after any restart, because the old code had no ban guard and the
        in-memory cooldown died with the previous process. Now:
          - ban active  -> disk cache (<= 7 days old) or static coins, and
                           ZERO network weight is spent (no 80w /ticker/24hr
                           into a banned IP, no ERROR spam - a ban is an
                           expected state, not a failure);
          - fetch OK    -> universe persisted to data/symbols_cache.json;
          - fetch fails -> previous fallback (static list), ERROR only when
                           no ban is active (a genuine network problem).
        """
        if _ban_active():
            cached = self._load_symbols_cache()
            if cached:
                log.info(
                    f"[cyan]Symbols[/] REST ban active - using disk-cached "
                    f"universe ({len(cached)} pairs, no REST request sent)"
                )
                return [s for s in cached if s not in self.excluded]
            static = [s for s in settings.load_coins() if s not in self.excluded]
            log.info(
                f"[cyan]Symbols[/] REST ban active and no cache yet - "
                f"starting from the static coins.yaml list "
                f"({len(static)} pairs, no REST request sent); the dynamic "
                f"universe loads after the ban lifts"
            )
            return static
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
            pairs = [t["symbol"] for t in sorted_pairs[:settings.MAX_SYMBOLS]]
            self._save_symbols_cache(pairs)  # v5.15: ban-safe boots later
            return pairs
        except Exception as e:
            if _ban_active():  # ban went active mid-request - expected, quiet
                log.warning(
                    f"Symbol fetch hit an active rate ban - keeping the "
                    f"current/static list ({e})"
                )
            else:
                log.error(f"Failed to fetch USDT pairs: {e}")
            # Fallback order: disk cache first (v5.15), then static list
            cached = self._load_symbols_cache()
            if cached:
                return [s for s in cached if s not in self.excluded]
            return [s for s in settings.load_coins() if s not in self.excluded]

    def refresh_symbols(self, force: bool = False):
        """Re-fetch the symbol list (dynamic mode only).

        v5.3: cached for SYMBOL_REFRESH_MIN minutes. The old behaviour
        refetched /ticker/24hr (weight 80!) every cycle AND again from
        bottom_scanner.scan() - pure waste when symbols rarely churn.
        v5.15: no-op while a REST ban is active - the current list is kept
        untouched (and _symbols_ts is NOT bumped, so the refresh happens
        automatically right after the ban lifts).
        """
        if not settings.USE_ALL_USDT_PAIRS:
            return
        if _ban_active():
            log.debug(
                "Symbol refresh skipped - REST ban active; will refresh "
                "after the cooldown"
            )
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

    def analyze_one(self, symbol: str, ws_only: bool = False) -> Dict:
        """Fetch data and run analysis for one symbol.

        v5.14: `ws_only=True` (degraded cycle during a REST ban) skips the
        order book fetch (weight 5) - candles come from the WS cache, so
        the symbol costs ZERO REST weight.
        """
        # v5.12: mid-burst ban - fail this symbol WITHOUT a doomed network
        # attempt and flag the reason so analyze_all can abort the burst.
        if _ban_active():
            return {"symbol": symbol, "skip": True, "reason": "rate ban"}
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

            # Fetch order book (only if not skipped for performance).
            # v5.14: ws_only mode skips it too - zero REST during a ban.
            order_book = None
            if not settings.SKIP_ORDER_BOOK and not ws_only:
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

    def analyze_all(self, parallel: bool = True, max_workers: int = None,
                    ws_only: bool = False) -> List[Dict]:
        """Analyze all configured symbols.

        v5.14: `ws_only=True` runs the burst purely off the WS candle cache
        (order books skipped) - used by the degraded cycle while a REST
        rate ban is active; WS streams bypass the REST budget entirely.
        """
        if max_workers is None:
            max_workers = settings.MAX_WORKERS
        log.info(
            f"[cyan]Starting market analysis[/] for {len(self.symbols)} symbols "
            f"(parallel={parallel}, workers={max_workers}, "
            f"timeframes={settings.TIMEFRAMES}, ws_only={ws_only})"
        )
        start = time.time()

        # Refresh symbols list before each cycle (when in dynamic mode).
        # v5.14: skipped in ws_only mode - /ticker/24hr (weight 80) is REST
        # and the cached list is good enough while a ban is active.
        if settings.USE_ALL_USDT_PAIRS and not ws_only:
            self.refresh_symbols()

        results = []
        ban_skips = 0  # v5.12: symbols skipped because a ban went active
        if parallel and len(self.symbols) > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(self.analyze_one, s, ws_only): s
                           for s in self.symbols}
                for fut in as_completed(futures):
                    try:
                        r = fut.result()
                        if r.get("reason") == "rate ban":
                            ban_skips += 1
                            continue
                        if not r.get("skip"):
                            results.append(r)
                    except Exception as e:
                        log.error(f"Future error: {e}")
                    # v5.12: a ban went active mid-burst - cancel the queued
                    # futures (they would all fail fast anyway) and stop now.
                    if ban_skips >= 3 and _ban_active():
                        for f in futures:
                            f.cancel()
                        break
        else:
            for s in self.symbols:
                r = self.analyze_one(s, ws_only=ws_only)
                if r.get("reason") == "rate ban":
                    ban_skips += 1
                    if ban_skips >= 3 and _ban_active():
                        break
                    continue
                if not r.get("skip"):
                    results.append(r)

        # v5.12: burst aborted by a rate ban -> keep the LAST GOOD snapshot.
        # Overwriting data/recommendations.json with an empty/parital run
        # used to blank the dashboard and trigger "No strong signals" while
        # the real problem was the ban, not the market.
        if ban_skips >= 3 or (ban_skips > 0 and _ban_active()):
            self.last_run_aborted = True
            log.warning(
                f"[yellow]Analysis burst ABORTED by Binance rate ban "
                f"({ban_skips} symbol(s) skipped) - last good recommendations "
                f"snapshot kept; positions stay guarded by the 1-min watcher[/]"
            )
            return []
        self.last_run_aborted = False

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

        # === BOOST MECHANISM: Integrate Bottom Scanner candidates (v5.6) ===
        # The bottom scanner finds coins at recent lows with strong bounce
        # signals. v5.6 fixes the two silent blockers that made this channel
        # dead (user report: "البوت لا يفتح توصيات من عملات القاع"):
        #   1) Admission no longer reuses the STRATEGY confidence scale:
        #      the bounce score has its own gate (BOTTOM_STRONG_SCORE) and
        #      its own mapping (40 + score/2, cap BOTTOM_CONF_CAP). Under the
        #      old formula (45 + score/3) + MIN_CONFIDENCE=68 the de-facto
        #      floor was score >= 69 - most candidates died here.
        #   2) Boosted recs now carry a bounce-derived "harmony" (layered
        #      bounce agreement) AND validate_recommendation exempts them
        #      from the strategy harmony gate - previously every boosted rec
        #      was rejected at open time with "Harmony too low (0.00)".
        # SAFETY RULES (kept):
        #   - Last candle closed bullish (no falling knives)
        #   - Max BOTTOM_MAX_PER_CYCLE boosted entries per cycle
        #   - Confidence capped at BOTTOM_CONF_CAP: genuine strategy
        #     signals still rank first
        #   - RR must clear MIN_RR_RATIO (bottom SL/TP = 1.2/2.5 ATR -> ~2.08)
        strong_bottoms = []
        try:
            if not settings.BOTTOM_BOOST_ENABLED:
                log.info("[cyan]Bottom boost disabled[/] (BOTTOM_BOOST_ENABLED=false)")
                raise StopIteration  # skip the whole boost block cleanly
            from src.analysis.bottom_scanner import bottom_scanner
            log.info("[cyan]Running bottom scanner for boost mechanism...[/]")
            bottom_candidates = bottom_scanner.scan(max_candidates=20, limit=200)
            strong_bottoms = [
                c for c in bottom_candidates
                if c.get("score", 0) >= settings.BOTTOM_STRONG_SCORE
                and c.get("last_candle_bullish", False)
            ]
            log.info(
                f"[green]{len(strong_bottoms)} confirmed bottom candidates[/] "
                f"(score >= {settings.BOTTOM_STRONG_SCORE:g} + bullish close)"
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
                if boosted >= settings.BOTTOM_MAX_PER_CYCLE:
                    break
                if c["symbol"] in existing_symbols:
                    continue
                rec = build_bottom_rec(c)
                # Boosted entries must clear the RR gate; their ADMISSION
                # gate is the bounce score (applied in strong_bottoms) - not
                # the strategy-scale MIN_CONFIDENCE.
                if rec["risk_reward_ratio"] >= settings.MIN_RR_RATIO:
                    filtered.append(rec)
                    boosted += 1
            if boosted:
                log.info(f"[green]{boosted} bottom candidates boosted into recommendations[/]")
                # v5.6: re-rank the merged list with the SAME composite key
                # filter_signals uses, so boosted recs compete for the top
                # MAX_RECOMMENDATIONS slots instead of being appended last
                # and silently cut by the [:MAX_RECOMMENDATIONS] slice.
                filtered.sort(
                    key=lambda r: (
                        r.get("confidence", 0)
                        + 5.0 * min(r.get("risk_reward_ratio", 0), 3.0) / 3.0
                        + 4.0 * min(r.get("harmony", 0.0), 1.0)
                    ),
                    reverse=True,
                )
        except StopIteration:
            pass
        except Exception as e:
            log.error(f"Bottom scanner boost failed: {e}")

        # Limit
        # v5.13: Regime Router - re-rank the merged candidate list by how
        # well each rec's strategies fit the CURRENT market state (leaders
        # + Fear & Greed + weekend + BTC volatility), then slice the top N.
        # In a ranging regime a mean-reversion candidate now outranks an
        # equal-scoring breakout candidate; in a crisis everything shrinks.
        try:
            if settings.REGIME_ENABLED:
                from src.analysis.regime_router import regime_router
                top_pre = filtered[:settings.MAX_RECOMMENDATIONS * 2]
                regime_router.apply_regime_routing(top_pre)
                filtered = top_pre
        except Exception as e:
            log.warning(f"[yellow]Regime routing skipped: {e}[/]")

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
        # v5.13: embed the regime snapshot so dashboard/WS read it free
        try:
            if settings.REGIME_ENABLED:
                from src.analysis.regime_router import regime_router
                rg = regime_router.get_regime()
                save_data["regime"] = {
                    k: rg.get(k) for k in
                    ("state", "state_ar", "weekend", "fear_greed",
                     "volatility", "leaders", "size_multiplier",
                     "min_confidence_adjust", "min_rr_adjust",
                     "allow_new_entries", "reasons_ar")
                }
        except Exception:
            pass
        save_json(to_json_safe(save_data), RECOMMENDATIONS_FILE)
        log.info(f"Recommendations saved to {RECOMMENDATIONS_FILE}")

        return top


# Singleton
analyzer = MarketAnalyzer()
