"""Micro-Scalp Scanner (v5.25) - the user's strategy channel on TRUE
15s/30s candles (resampled from 1s klines), long-only, fixed 1-minute hold.

Pipeline (mirrors the v5.22 momentum channel):

  dedicated 1-min job (both hosts) -> scalp_scanner.tick()
      -> scan(liquid head, 1s klines -> MicroScalpStrategy 15s/30s)
      -> 100% checklist candidates -> build_scalp_rec
      -> risk_manager.open_position (channel caps) -> Telegram notify

Cost discipline (the bot lives on a shared Render IP with a 418 history):
  * ban guard: zero REST calls while a 429/418 cooldown is active;
  * WS-seeding guard: the recovery windows belong to the paced seeder -
    the scalper skips a tick while seeding is still running;
  * TOP-N shortlist (15) x weight-5 (1s klines, limit 1000) = 75 REST
    weight per tick - a rounding error against the 4500/min budget;
  * RateLimitError aborts the whole tick (every following call would fail
    for the same reason);
  * per-symbol cooldown stops the same burst from re-firing while the same
    3-candle episode is still running.
"""
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

from config.settings import settings
from src.core.data_fetcher import data_fetcher
from src.core.rate_limiter import RateLimitError
from src.strategies.micro_scalp_strategy import MicroScalpStrategy
from src.utils.helpers import now_utc, save_json, to_json_safe
from src.utils.logger import log

SCALP_SIGNALS_FILE = Path("data/scalp_signals.json")


class ScalpScanner:
    """Scans true 15s/30s candles (from 1s klines) for the micro-scalp
    strategy: BB 11/3 + SuperTrend 2/2, long-only, fixed 60s holding.

    v5.29: the shortlist is ranked by the WS live-price move over the
    speed window (movers first) BEFORE truncating to SCALP_TOP_N - the
    1s-klines REST weight goes to symbols that are actually moving instead
    of a static alphabetical head, cutting the channel's REST footprint
    (15 x weight-5 per tick was the bot's largest recurring REST cost)
    while IMPROVING candidate quality (the checklist needs bursts).
    """

    def __init__(self):
        self._strategy = MicroScalpStrategy()
        # {symbol: ts of last fired signal} - burst dedup across ticks
        self._last_fired: Dict[str, float] = {}
        self._last_tick_stats: Dict = {}
        # v5.29: zero-REST momentum ring - {symbol: deque[(ts, price)]}
        # sampled from the WS live-price cache every tick (even when a rate
        # ban or a seeding window skips the scan itself).
        self._price_ring: Dict[str, deque] = {}
        self._ring_window_s = float(max(120, settings.SCALP_SPEED_WINDOW_S))
        self._ring_maxlen = max(8, int(self._ring_window_s / 30.0))
        log.info(
            "[cyan]ScalpScanner[/] initialized (micro_scalp: "
            f"BB {settings.SCALP_BB_PERIOD}/{settings.SCALP_BB_DEV:g} "
            f"+ SuperTrend {settings.SCALP_ST_PERIOD}/"
            f"{settings.SCALP_ST_MULT:g} on true "
            f"{settings.SCALP_TF_CALM}s/{settings.SCALP_TF_FAST}s candles "
            f"resampled from 1s, long-only, fixed "
            f"{settings.SCALP_HOLD_SECONDS}s hold, WS-mover shortlist)")

    # ------------------------------------------------------------------
    def _symbol_on_cooldown(self, symbol: str) -> bool:
        ts = self._last_fired.get(symbol)
        if ts is None:
            return False
        return (time.time() - ts) < settings.SCALP_SYMBOL_COOLDOWN_MIN * 60.0

    def _ws_seeding(self) -> bool:
        """True while the WS paced seeder is still filling the cache - the
        REST recovery windows belong to it (the v5.24 lesson)."""
        try:
            from src.core.ws_feed import ws_feed
            st = ws_feed.cache_status() or {}
            return bool(st.get("seeding"))
        except Exception:
            return False

    # ------------------------------------------------------------------
    # v5.29: zero-REST momentum ring (WS live-price samples per tick)
    # ------------------------------------------------------------------
    def _sample_prices(self, symbols: List[str]) -> None:
        """Sample the WS live-price cache into the per-symbol ring.

        Pure in-memory: never fires a REST call, never raises, and keeps
        sampling even while a rate ban or a seeding window skips the scan -
        so the ring is already warm when scanning resumes.
        """
        try:
            from src.core.ws_feed import ws_feed
            prices = ws_feed.get_live_prices(symbols, max_age_s=180.0)
        except Exception:
            return
        now = time.time()
        for sym, price in (prices or {}).items():
            try:
                p = float(price)
            except (TypeError, ValueError):
                continue
            if p <= 0:
                continue
            ring = self._price_ring.setdefault(
                sym, deque(maxlen=self._ring_maxlen))
            ring.append((now, p))
        # bound the dict (delisted symbols never return prices again)
        if len(self._price_ring) > 400:
            for sym in list(self._price_ring):
                if sym not in prices:
                    self._price_ring.pop(sym, None)

    def _move_pct(self, symbol: str) -> Optional[float]:
        """Signed % move over the ring window (None when coverage is thin).

        Requires >= 2 samples spanning >= half the window - a cold process
        or a stale WS feed yields None and the symbol keeps legacy order.
        """
        ring = self._price_ring.get(symbol)
        if not ring:
            return None
        now = time.time()
        horizon = now - self._ring_window_s
        fresh = [(ts, p) for (ts, p) in ring if ts >= horizon]
        if len(fresh) < 2:
            return None
        span = fresh[-1][0] - fresh[0][0]
        if span < self._ring_window_s * 0.5:
            return None
        p0, p1 = fresh[0][1], fresh[-1][1]
        if p0 <= 0:
            return None
        return (p1 - p0) / p0 * 100.0

    def _rank_shortlist(self, shortlist: List[str]) -> List[str]:
        """Movers first (WS price move over the speed window).

        LONG-ONLY channel: a dumping symbol can never fire the green-run
        checklist, so negative movers rank below flat/unknown ones and only
        POSITIVE movers jump the queue. With no ring data (cold boot, WS
        down) the original order is kept unchanged.
        """
        known = []
        unknown = []
        for sym in shortlist:
            mv = self._move_pct(sym)
            if mv is None:
                unknown.append(sym)
            else:
                known.append((max(mv, 0.0), sym))
        if not known:
            return shortlist
        known.sort(key=lambda t: -t[0])  # stable sort: ties keep order
        return [sym for _, sym in known] + unknown

    # ------------------------------------------------------------------
    def scan(self, symbols: Optional[List[str]] = None,
             max_candidates: Optional[int] = None) -> List[Dict]:
        """Scan the liquid shortlist; return micro-scalp candidates."""
        if not settings.SCALP_ENABLED:
            return []
        # v5.12 pattern: a hard 429/418 ban makes every REST klines call
        # here doomed - bail out cleanly, the next tick retries.
        try:
            from src.core.rate_limiter import rate_limiter
            if rate_limiter.cooldown_remaining() > 130.0:
                log.warning(
                    f"[yellow]Scalp scanner skipped[/] - Binance rate "
                    f"ban active ({rate_limiter.cooldown_remaining():.0f}s "
                    f"left)")
                return []
        except Exception:
            pass
        if self._ws_seeding():
            log.debug("Scalp scan skipped - WS seeder owns the REST window")
            return []

        if symbols is None:
            from src.analysis.analyzer import analyzer
            symbols = analyzer.symbols
        shortlist = [s for s in (symbols or [])
                     if not self._symbol_on_cooldown(s)]
        # v5.29: rank by the WS price move BEFORE truncating - REST weight
        # goes to symbols actually moving, not to a static list head.
        ranked = self._rank_shortlist(shortlist)
        shortlist = ranked[:max(1, settings.SCALP_TOP_N)]
        if not shortlist:
            return []

        n_movers = sum(1 for s in shortlist if self._move_pct(s) is not None)
        log.info(
            f"[cyan]Scalp scanning[/] {len(shortlist)} symbols "
            f"({n_movers} WS-ranked movers) on true "
            f"{settings.SCALP_TF_CALM}s/{settings.SCALP_TF_FAST}s candles "
            f"(1s klines, micro_scalp)...")
        start = time.time()

        candidates: List[Dict] = []
        rate_aborted = False
        for symbol in shortlist:
            try:
                c = self._scan_one(symbol)
            except Exception as e:
                # A rate refusal here means every following call would fail
                # for the same reason (same shared IP) - stop the tick.
                if isinstance(e, RateLimitError):
                    log.warning(
                        f"[yellow]Scalp tick aborted[/] - rate budget "
                        f"refused ({e})")
                    rate_aborted = True
                    break
                log.debug(f"Scalp scan error {symbol}: {e}")
                continue
            if c and not c.get("skip"):
                candidates.append(c)

        candidates.sort(key=lambda c: c.get("score", 0), reverse=True)
        top = candidates[:max(1, max_candidates or settings.SCALP_MAX_PER_TICK)]

        self._last_tick_stats = {
            "symbols_scanned": len(shortlist),
            "candidates_found": len(candidates),
            "rate_aborted": rate_aborted,
            "scan_time_seconds": round(time.time() - start, 2),
        }
        save_json(to_json_safe({
            "timestamp": now_utc().isoformat(),
            "timeframes": f"{settings.SCALP_TF_CALM}s/{settings.SCALP_TF_FAST}s"
            " (from 1s)",
            "symbols_scanned": len(shortlist),
            "candidates_found": len(candidates),
            "rate_aborted": rate_aborted,
            "top_candidates": top,
            "scan_time_seconds": round(time.time() - start, 2),
        }), SCALP_SIGNALS_FILE)

        if top:
            log.info(
                f"[green]Scalp scan[/] - {len(top)} micro-scalp candidate(s) "
                f"in {time.time()-start:.1f}s")
        return top

    def _scan_one(self, symbol: str) -> Optional[Dict]:
        df = data_fetcher.get_candles(
            symbol, "1s", limit=settings.SCALP_KLINES_LIMIT)
        if df is None or len(df) == 0:
            return {"symbol": symbol, "skip": True,
                    "reason": "no 1s data"}
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
            "timeframe": d.get("timeframe", "15s"),
            "current_price": price,
            "stop_loss": sl,
            "take_profit": tp1,
            "take_profit_2": tp2,
            "risk_reward_ratio": float(rr),
            "atr": atr_val,
            "atr_pct": float(atr_val / price * 100) if price > 0 else 0.0,
            "near_upper_dist_pct": d.get("near_upper_dist_pct"),
            "volume_ratio": d.get("volume_ratio"),
            "st_line": d.get("st_line"),
            "st_trend": d.get("st_trend"),
            "signals": sig.reasons,
            "green_run": d.get("green_run"),
            "analyzed_at": now_utc().isoformat(),
        }

    # ------------------------------------------------------------------
    def tick(self) -> int:
        """One scheduler tick: scan -> open (with session gate) -> notify.
        Returns the number of positions opened."""
        if not settings.SCALP_ENABLED:
            return 0
        # v5.29: sample WS prices BEFORE any skip - the momentum ring must
        # keep filling even while a rate ban or a seeding window skips the
        # scan, so the shortlist is already warm when scanning resumes.
        try:
            from src.analysis.analyzer import analyzer
            self._sample_prices(analyzer.symbols)
        except Exception:
            pass
        candidates = self.scan()
        if not candidates:
            return 0

        opened = 0
        for c in candidates:
            if opened >= max(1, settings.SCALP_MAX_PER_TICK):
                break
            # Session gate (same hard NEW-ENTRY rules as every channel).
            try:
                from src.analysis.session_clock import entry_gate, session_info
                _blocked, _why = entry_gate(
                    session_info(), "micro_scalp", False)
                if _blocked:
                    log.info(
                        f"[yellow]Scalp entry blocked[/] ({c['symbol']}): "
                        f"{_why}")
                    continue
            except Exception:
                pass  # the session layer may only skip, never crash

            rec = build_scalp_rec(c)
            from src.risk.manager import risk_manager
            res = risk_manager.open_position(rec)
            if res.get("status") != "opened":
                log.debug(
                    f"Scalp entry rejected {rec['symbol']}: "
                    f"{res.get('reasons')}")
                continue
            opened += 1
            pos = res.get("position", {})
            log.info(
                f"[green]SCALP opened[/] {rec['symbol']} on "
                f"{c.get('timeframe')} - hold fixed "
                f"{settings.SCALP_HOLD_SECONDS}s")
            try:
                from src.core.notifier import telegram_notifier
                if telegram_notifier.enabled:
                    telegram_notifier.send_alert(
                        f"سكالب دقيقة 🪒 {rec['symbol']} "
                        f"({c.get('timeframe')} من شموع 1 ثانية)",
                        f"دخول: {rec['current_price']}\n"
                        f"وقف امان: {rec['stop_loss']}\n"
                        f"هدف جزئي: {rec['take_profit']} | "
                        f"هدف ممتد: {rec['take_profit_2']}\n"
                        f"المدة ثابتة: {settings.SCALP_HOLD_SECONDS} ثانية "
                        f"(خروج زمني)\n"
                        f"الشروط: {len(c.get('signals') or [])} تحقق كامل "
                        f"(3 شموع خضراء فوق SuperTrend قرب الحد العلوي)"
                    )
            except Exception:
                pass
        return opened


def build_scalp_rec(c: Dict) -> Dict:
    """v5.25: convert a scalp-scanner candidate into a recommendation dict.

    Same contract as build_double_ind_rec: the entry conditions are binary
    (the strategy's 100% checklist), so the candidate score IS the
    confidence. boosted_from_scalp=True exempts the rec from the
    strategy-scale MIN_HARMONY gate (same contract as bottom/momentum) and
    activates the channel caps in risk manager (_scalp_channel_ok).
    """
    score = float(c.get("score", 0))
    confidence = max(0.0, min(score, settings.SCALP_CONF_CAP))
    n_layers = len(c.get("signals") or [])
    harmony = round(min(0.85, 0.35 + 0.10 * max(0, n_layers - 1)), 2)
    price = float(c["current_price"])
    tp2 = float(c.get("take_profit_2") or c.get("take_profit"))
    return {
        "symbol": c["symbol"],
        "direction": "bullish",  # LONG-ONLY: the channel has no sell branch
        "weighted_score": score,
        "confidence": confidence,
        "admission_confidence": confidence,
        "current_price": price,
        # Expected move = the runner target (TP2); its floor (1.6%) keeps
        # this above MIN_EXPECTED_RISE by construction.
        "expected_rise_pct": max(
            settings.MIN_EXPECTED_RISE,
            abs(tp2 - price) / max(price, 1e-12) * 100.0),
        "stop_loss": c["stop_loss"],
        "take_profit": c["take_profit"],
        "take_profit_2": tp2,
        "risk_reward_ratio": float(c.get("risk_reward_ratio", 0.0)),
        "atr": float(c.get("atr") or 0.0),
        "atr_pct": float(c.get("atr_pct") or 0.0),
        "harmony": harmony,
        # Scalp entries buy strength at market - no limit zone to chase.
        "entry_type": "market",
        "signals": [{
            "strategy": "micro_scalp",
            "direction": "bullish",
            "score": score,
            "confidence": confidence / 100.0,
            "reasons": c.get("signals", []),
            "details": {
                "timeframe": c.get("timeframe"),
                "near_upper_dist_pct": c.get("near_upper_dist_pct"),
                "volume_ratio": c.get("volume_ratio"),
                "green_run": c.get("green_run"),
                "st_line": c.get("st_line"),
            },
        }],
        "timeframe": c.get("timeframe", "15s"),
        "analyzed_at": c.get("analyzed_at"),
        "boosted_from_scalp": True,
    }


# Singleton
scalp_scanner = ScalpScanner()
