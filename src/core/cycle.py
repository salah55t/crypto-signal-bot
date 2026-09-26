"""
Core trading cycle (v5) - THE single source of truth for bot behaviour.

Previously the web dashboard (src/web/app.py) and the standalone runner
(scripts/run_bot.py) each had their OWN copy of the trade-management and
position-opening logic, which drifted apart (the web copy missed several
v4.1 gates). This module unifies everything:

  manage_open_positions()   SL/TP + TP1 partial + time stop + chandelier
                            trailing + structural (Ichimoku) exits
  open_new_positions()      v4.1 gates + harmony + market-tide + pending
                            LIMIT entries (no chasing)
  run_position_watch()      1-minute real-time watcher: price-only checks,
                            pending fills, excursions (near-zero API weight)
  run_analysis_cycle()      full cycle: manage -> analyze -> manage with
                            signals -> notify -> open

Both run_bot.py and the web app now delegate here.
"""
import threading
import time
from pathlib import Path
from typing import Dict, List, Tuple

from config.settings import settings
from src.core.binance_client import binance_client
from src.core.data_fetcher import data_fetcher
from src.risk.manager import risk_manager
from src.db.database import db
from src.notifications import telegram_notifier, file_logger
from src.utils.logger import log
from src.utils.helpers import load_json, save_json, to_json_safe, now_utc
from src.utils.i18n import tr, ar_mode, ar_entry_type

DATA_DIR = Path("data")
CLOSED_TRADES_FILE = DATA_DIR / "closed_trades.json"
RECOMMENDATIONS_FILE = DATA_DIR / "recommendations.json"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def fetch_prices(symbols, priority: bool = False) -> Dict[str, float]:
    """v5: batched price fetch - ONE request (weight 2-4), not 80.
    v5.10: priority=True uses the reserved lane of the rate limiter
    (position watch / dashboard must not be starved by bulk analysis).
    """
    if not symbols:
        return {}
    try:
        return data_fetcher.get_batch_prices(symbols, priority=priority)
    except Exception as e:
        log.error(f"Batch price fetch failed: {e}")
        return {}


def _market_signals_from_file() -> Dict[str, Dict]:
    """Latest full analysis results keyed by symbol (for exits/trailing)."""
    try:
        full = load_json(RECOMMENDATIONS_FILE, default={})
        return {
            r["symbol"]: r for r in full.get("all_results", [])
            if isinstance(r, dict) and "symbol" in r
        }
    except Exception:
        return {}


def _save_closed_trades(closed: list):
    if not closed:
        return
    try:
        existing = load_json(CLOSED_TRADES_FILE, default=[])
        existing.extend(closed)
        save_json(to_json_safe(existing), CLOSED_TRADES_FILE)
    except Exception as e:
        log.error(f"Failed to save closed trades: {e}")


def _autopsy_lines(c: Dict) -> List[str]:
    """v5.11 trade autopsy - the numbers a veteran asks about after a close.

    Rendered in the Telegram close card AND fed to the AI lesson:
    whole-trade P&L (partials included), how far the price travelled in our
    favour (MFE), how deep it hurt (MAE), and how much of the peak we kept.
    """
    lines = []
    total = c.get("total_pnl")
    if total is not None and abs(float(total) - float(c.get("pnl", 0) or 0)) > 1e-9:
        lines.append(
            f"إجمالي الصفقة (شامل جني TP1): ${float(total):+.2f} "
            f"({float(c.get('total_pnl_pct', 0) or 0):+.2f}%)"
        )
    if c.get("mfe_pct"):
        lines.append(f"أعلى ربح مرّ به السعر: +{float(c['mfe_pct']):.2f}%")
    if c.get("mae_pct"):
        lines.append(f"أعمق تراجع: -{float(c['mae_pct']):.2f}%")
    if c.get("capture_efficiency") is not None:
        lines.append(
            f"كفاءة التقاط القمة: {float(c['capture_efficiency']):.0f}%")
    if c.get("duration_hours") is not None:
        lines.append(f"مدة الصفقة: {float(c['duration_hours']):.1f} ساعة")
    if c.get("partials_count"):
        lines.append(f"جني جزئي TP1: {int(c['partials_count'])} مرة")
    if c.get("risk_updates_count"):
        lines.append(f"تعديلات وقف/هدف: {int(c['risk_updates_count'])}")
    return lines


def _send_ai_lesson_async(c: Dict):
    """v5.11 creative move: an AI post-mortem lesson for every closed trade.

    Deliberately OFF the critical path - a daemon thread calls the LLM and
    sends the lesson as a follow-up Telegram message, so a slow model can
    never delay SL/TP checks. Any failure = no lesson, silently.
    """
    if not telegram_notifier.enabled:
        return

    def _work():
        try:
            from src.ai import ai_advisor
            lesson = ai_advisor.trade_postmortem(c)
            if lesson:
                telegram_notifier.send_alert(
                    f"درس الصفقة 🧠 {c.get('symbol', '')}",
                    lesson,
                )
        except Exception as e:
            log.debug(f"AI lesson skipped: {e}")

    threading.Thread(
        target=_work, daemon=True,
        name=f"ai-lesson-{c.get('symbol', 'x')}").start()


def _notify_closed(closed: list):
    for c in closed:
        if not telegram_notifier.enabled:
            continue
        if c.get("status") == "partial":
            telegram_notifier.send_alert(
                f"جني جزئي TP1 🎯 {c.get('symbol', '')}",
                f"تم تحقيق {c.get('fraction', 0)*100:.0f}% من الصفقة عند {c.get('exit_price')}\n"
                f"الربح/الخسارة: ${c.get('pnl', 0):+.2f} ({c.get('pnl_pct', 0):+.2f}%)\n"
                f"المحقق تراكمياً: ${c.get('realized_pnl', 0):+.2f}\n"
                f"الكمية المتبقية تستهدف TP2 ووقفها عند نقطة التعادل"
            )
            continue
        win_emoji = "✅" if c.get("total_pnl", c.get("pnl", 0)) > 0 else "❌"
        autopsy = _autopsy_lines(c)
        body = (
            f"العملة: {c.get('symbol')}\n"
            f"الدخول: {c.get('entry_price')}\n"
            f"الخروج: {c.get('exit_price')}\n"
            f"ربح هذا الجزء: ${c.get('pnl', 0):+.2f} ({c.get('pnl_pct', 0):+.2f}%)\n"
        )
        if autopsy:
            body += "\n".join(autopsy) + "\n"
        body += (
            f"السبب: {tr(c.get('reason', ''))}\n"
            f"الوضع: {ar_mode('PAPER' if c.get('paper', True) else 'LIVE')}"
        )
        telegram_notifier.send_alert(f"إغلاق صفقة {win_emoji}", body)
        # v5.11: the AI lesson arrives seconds later as its own message
        _send_ai_lesson_async(c)


def _notify_updates(updates: list, tag: str = ""):
    tag_ar = " (مراقبة)" if tag.strip() == "(watch)" else (f" {tag}" if tag else "")
    for u in updates:
        if telegram_notifier.enabled:
            telegram_notifier.send_alert(
                f"تحديث مخاطر 🔧 {u.get('symbol', '')}{tag_ar}",
                f"{tr(u.get('reason', ''))}\n"
                f"وقف قديم: {u.get('old_sl')} → وقف جديد: {u.get('new_sl')}\n"
                f"هدف قديم: {u.get('old_tp')} → هدف جديد: {u.get('new_tp')}"
            )


# ------------------------------------------------------------------
# STEP 1: manage existing positions
# ------------------------------------------------------------------
def manage_open_positions(market_signals: Dict[str, Dict] = None,
                          structural: bool = True) -> Tuple[List[Dict], List[Dict]]:
    """
    Full veteran management pass over open positions.

    market_signals: latest per-symbol analysis (needed for structural exits
                    + chandelier ATR). When None it is loaded from the last
                    recommendations.json snapshot.
    structural:     False inside the 1-min watcher (no fresh klines there).
    Returns (closed_or_partial_results, sl_tp_updates).
    """
    market_signals = market_signals if market_signals is not None \
        else _market_signals_from_file()
    if not risk_manager.open_positions:
        return ([], [])

    pos_symbols = list({p["symbol"] for p in risk_manager.open_positions})
    # v5.10: priority lane - SL/TP checks on open positions are exactly the
    # traffic the reserved budget exists for; add the last-known fallback so
    # a pressure cooldown does not leave positions unmanaged.
    current_prices = fetch_prices(pos_symbols, priority=True)
    if not current_prices:
        current_prices = data_fetcher.get_last_known_prices(
            pos_symbols, max_age_s=120)
    if not current_prices:
        log.warning("Could not fetch prices for open positions - skipping management")
        return ([], [])

    # 1) SL / TP1-partial / TP2 / time-stop (price-based)
    results = risk_manager.check_open_positions(current_prices)
    for r in results:
        if r.get("status") == "partial":
            log.info(
                f"[green]TP1 partial[/] {r['position']['symbol']} - "
                f"${r.get('pnl', 0):+.2f}"
            )
        else:
            log.info(
                f"[yellow]Position closed[/] {r['symbol']} - "
                f"PnL: ${r.get('pnl', 0):+.2f} ({r.get('pnl_pct', 0):+.2f}%) "
                f"- {r.get('reason', '')}"
            )
    _save_closed_trades([r for r in results if r.get("status") != "partial"])
    _notify_closed(results)

    # 2) Structure-aware exits (Ichimoku flip / opposite signal) - main cycle
    if structural and settings.STRUCTURAL_EXITS_ENABLED:
        for i in range(len(risk_manager.open_positions) - 1, -1, -1):
            pos = risk_manager.open_positions[i]
            sig = market_signals.get(pos["symbol"])
            price = current_prices.get(pos["symbol"])
            if not sig or not price:
                continue
            action, reason = risk_manager.evaluate_structural_exit(pos, sig, price)
            if action == "exit":
                closed = risk_manager.close_position(i, price,
                                                     f"Structural exit: {reason}")
                _save_closed_trades([closed])
                _notify_closed([closed])
                log.info(
                    f"[red]Structural exit[/] {pos['symbol']} - {reason}"
                )
            elif action == "tighten":
                # pull the structural level out of the reason text
                level = None
                try:
                    level = float(reason.rsplit("(", 1)[1].rstrip(")").split(" ")[0]
                                  .replace(",", ""))
                except Exception:
                    level = None
                if level:
                    res = risk_manager.update_position_risk(
                        i, price, level, None, f"Structural: {reason}")
                    if res.get("status") == "updated":
                        log.info(f"[blue]Structural tighten[/] {pos['symbol']} - {reason}")

    # 3) Chandelier + ladder trailing on the survivors
    updates = []
    if risk_manager.open_positions:
        updates = risk_manager.apply_trailing_logic(current_prices, market_signals)
        for u in updates:
            log.info(f"[blue]Risk update[/] {u.get('reason', '')}")
    _notify_updates(updates)

    return (results, updates)


# ------------------------------------------------------------------
# STEP 2 (watcher): 1-minute real-time position watch + pending fills
# ------------------------------------------------------------------
def run_position_watch():
    """
    Real-time tracking loop (every 1 minute, price-only):
      - SL / TP1-partial / TP2 / time-stop checks
      - MFE/MAE excursion tracking
      - pending LIMIT entry fills (buy the pocket when price returns)
    Weight cost: ONE batched price request (weight 2-4) per minute.
    """
    try:
        has_positions = bool(risk_manager.open_positions)
        has_pending = bool(risk_manager.pending_entries)
        if not has_positions and not has_pending:
            return

        symbols = list({
            p["symbol"] for p in risk_manager.open_positions
        } | {p["symbol"] for p in risk_manager.pending_entries})
        # v5.10: priority lane first; if even that fails (shared-IP cooldown),
        # very-fresh last-known prices (<= 120s) keep SL/TP checking alive
        # instead of going blind for the whole cooldown window.
        prices = fetch_prices(symbols, priority=True)
        if not prices:
            prices = data_fetcher.get_last_known_prices(symbols, max_age_s=120)
            if prices:
                log.info(
                    f"[yellow]Position watch:[/] live fetch unavailable - "
                    f"using last-known prices for {len(prices)} symbol(s)")
        if not prices:
            return

        # 1) pending limit fills (price returned to the entry zone)
        if risk_manager.pending_entries:
            filled = risk_manager.check_pending_fills(prices)
            for f in filled:
                if telegram_notifier.enabled:
                    pos = f.get("position", {})
                    telegram_notifier.send_alert(
                        f"تنفيذ أمر دخول معلق 🎯 {f.get('symbol', '')}",
                        f"نُفذ عند: {f.get('fill_price')}\n"
                        f"تم احترام منطقة الدخول (بدون مطاردة السعر)\n"
                        f"وقف: {pos.get('stop_loss')} | هدف: {pos.get('take_profit')}"
                    )

        # 2) price-only position management (no structural analysis here)
        if risk_manager.open_positions:
            results = risk_manager.check_open_positions(prices)
            _save_closed_trades([r for r in results if r.get("status") != "partial"])
            _notify_closed(results)

            # light trailing pass with the cached signals (chandelier)
            updates = risk_manager.apply_trailing_logic(prices)
            _notify_updates(updates, tag="(watch)")
    except Exception as e:
        log.debug(f"Position watch error: {e}")


# ------------------------------------------------------------------
# STEP 3: open new positions (all veteran gates)
# ------------------------------------------------------------------
def open_new_positions(recommendations: List[Dict]) -> int:
    """Apply every admission gate, then open (or arm pending) positions."""
    if not recommendations:
        return 0

    # v5.13: regime freeze gate - in a full crisis (high BTC volatility +
    # bearish leaders + extreme fear) the router freezes NEW entries
    # entirely; open positions are still managed normally.
    try:
        if settings.REGIME_ENABLED:
            from src.analysis.regime_router import regime_router
            pol = regime_router.active_policy()
            if not pol.get("allow_new_entries", True):
                log.warning(
                    f"[yellow]Regime freeze gate:[/] "
                    f"{pol.get('freeze_reason', 'crisis')} - "
                    f"no new entries this cycle"
                )
                return 0
    except Exception:
        pass

    # v4.1: sync today's opened count from DB (survives restarts on Render)
    try:
        today_rows = db.get_daily_stats(1)
        if today_rows and today_rows[0].get("date") == risk_manager._today_key():
            risk_manager.sync_daily_opened(int(today_rows[0].get("trades_opened") or 0))
    except Exception as e:
        log.debug(f"Daily opened sync skipped: {e}")

    # v5: market tide gate (BTC regime) - blocks NEW entries only
    tide_blocked, tide_reason = risk_manager.market_tide_blocked()
    if tide_blocked:
        log.warning(f"[yellow]Market tide gate:[/] {tide_reason}")

    open_symbols = {p["symbol"] for p in risk_manager.open_positions}
    pending_symbols = {p["symbol"] for p in risk_manager.pending_entries}
    opened = 0

    for rec in recommendations:
        # v4.1 global gates: max positions / daily loss / trade cap / streak
        if not risk_manager.can_open_position():
            log.warning(
                "Risk gate (global): max positions / daily loss / trade cap / "
                "loss-streak pause"
            )
            break
        symbol = rec.get("symbol")
        # v4.1 per-symbol re-entry cooldown after a losing close
        if risk_manager.is_symbol_blocked(symbol):
            log.warning(f"[yellow]Skip {symbol}[/] - re-entry cooldown after recent loss")
            continue
        # duplicate protection (positions AND pending)
        if settings.SKIP_DUPLICATE_SYMBOLS and symbol in open_symbols:
            log.info(f"[yellow]Skip {symbol}[/] - position already open")
            continue
        if symbol in pending_symbols:
            log.info(f"[yellow]Skip {symbol}[/] - pending limit entry already armed")
            continue
        if tide_blocked:
            continue

        # v5 veteran entry discipline: LIMIT entries far above the golden
        # pocket are NOT chased - a pending order is armed instead.
        # Momentum bypass: A+ / very high confidence setups are the ONLY
        # ones allowed to enter at market above the zone.
        if (settings.PENDING_ENTRIES_ENABLED
                and rec.get("entry_type") == "limit"):
            zone = rec.get("entry_zone") or {}
            zone_high = zone.get("high")
            price = rec.get("current_price") or 0
            atr = rec.get("atr") or 0
            strong_momentum = (
                rec.get("a_plus", False)
                or float(rec.get("admission_confidence",
                                 rec.get("confidence", 0)) or 0)
                >= settings.PENDING_MOMENTUM_CONF
            )
            if (zone_high and price and atr
                    and rec.get("direction") == "bullish"
                    and price > float(zone_high) + settings.PENDING_CHASE_ATR * atr
                    and not strong_momentum):
                pending = risk_manager.add_pending_entry(
                    rec, "price above entry zone - waiting for pullback")
                pending_symbols.add(symbol)
                if telegram_notifier.enabled:
                    telegram_notifier.send_alert(
                        f"أمر دخول معلق ⏳ {symbol}",
                        f"منطقة الدخول: {pending['zone_low']:.4f} - "
                        f"{pending['zone_high']:.4f}\n"
                        f"الحالي: {price:.4f} (بانتظار ارتداد السعر للمنطقة)\n"
                        f"وقف: {rec.get('stop_loss')} | TP1: {rec.get('take_profit')} "
                        f"| TP2: {rec.get('take_profit_2')}\n"
                        f"ينتهي خلال {settings.PENDING_TTL_HOURS:.0f} ساعة"
                    )
                continue

        result = risk_manager.open_position(rec)  # auto paper/live
        if result.get("status") == "opened":
            opened += 1
            open_symbols.add(symbol)
            mode_tag = "PAPER" if settings.RUN_MODE == "paper" else "LIVE"
            log.info(f"[green]{mode_tag} position opened for {symbol}[/]")
            if telegram_notifier.enabled:
                pos = result.get("position", {})
                telegram_notifier.send_alert(
                    f"فتح صفقة 🚀 {symbol}",
                    f"الوضع: {ar_mode(mode_tag)}\n"
                    f"الدخول: {pos.get('entry_price')} "
                    f"({ar_entry_type(pos.get('entry_type', 'market'))})\n"
                    f"وقف الخسارة: {pos.get('stop_loss')}\n"
                    f"TP1: {pos.get('take_profit')} (يحقق 50% من الصفقة)\n"
                    f"TP2: {pos.get('take_profit_2')} (الكمية المتبقية)\n"
                    f"الثقة: {rec.get('confidence', 0):.0f}% | "
                    f"الانسجام: {rec.get('harmony', 0):.2f}"
                )
        else:
            log.warning(
                f"Position rejected: {result.get('reasons', result.get('reason'))}"
            )
    return opened


# ------------------------------------------------------------------
# FULL CYCLE
# ------------------------------------------------------------------
def rate_limit_gate() -> str:
    """v5.14 three-way gate: "run" | "degraded" | "skip".

    - "run"      : no cooldown - normal cycle (REST available).
    - "degraded" : a 429/418 or shared-IP cooldown is active BUT the WS
                   feed is connected and covers the universe - run the
                   cycle ENTIRELY off the live WS cache (zero REST weight;
                   REST bans do not touch the WS service).
    - "skip"     : cooldown active and the WS cache cannot cover the
                   universe - idle until the next cron tick (old behavior).
    """
    from src.core.rate_limiter import rate_limiter
    remaining = rate_limiter.cooldown_remaining()
    if remaining <= 0:
        return "run"
    if settings.USE_WS_FEED:
        try:
            from src.core.ws_feed import ws_feed
            if ws_feed.is_live():
                cov = ws_feed.coverage()
                if cov >= max(0.0, settings.WS_DEGRADED_COVERAGE):
                    log.warning(
                        f"[yellow]Rate-limit cooldown active "
                        f"({remaining:.0f}s left)[/] - [cyan]running WS-only "
                        f"cycle[/] (coverage {cov:.0%}, zero REST weight - "
                        f"WebSocket streams bypass the REST ban)"
                    )
                    return "degraded"
                log.warning(
                    f"[yellow]Rate-limit cooldown active ({remaining:.0f}s left)[/] - "
                    f"skipping this cycle (WS coverage {cov:.0%} < "
                    f"{settings.WS_DEGRADED_COVERAGE:.0%}); next cron tick retries"
                )
                return "skip"
        except Exception:
            pass
    log.warning(
        f"[yellow]Rate-limit cooldown active ({remaining:.0f}s left) - "
        f"skipping this cycle (429/418 or shared-IP pressure); "
        f"next cron tick retries automatically[/]"
    )
    return "skip"


def _binance_reachable(retries: int = 2, grace: float = 5.0) -> bool:
    """Ping with a short grace retry for transient network blips."""
    for attempt in range(1, retries + 1):
        if binance_client.ping():
            return True
        if attempt < retries:
            log.warning(f"Binance ping failed (attempt {attempt}) - retrying in {grace:.0f}s")
            time.sleep(grace)
    return False


def run_analysis_cycle():
    """One full bot cycle: manage -> analyze -> manage(signals) -> notify -> open.

    v5.14: during a REST rate ban the cycle still runs ("degraded") when
    the WS feed covers the universe - candles AND prices both come from
    WebSocket streams that Binance does not count against the REST budget.
    """
    log.info("=" * 60)
    log.info("[bold cyan]STARTING ANALYSIS CYCLE (v5)[/]")
    log.info("=" * 60)

    # ---- STEP 0: rate-limit gate (429/418 or shared-IP pressure) ----
    gate = rate_limit_gate()
    ws_only = (gate == "degraded")
    if gate == "skip":
        return

    # v5.14: a REST ping during a ban is exactly the poke we must avoid;
    # in degraded mode the whole cycle is WS-fed and needs no reachability
    # check (WS connectivity is verified by ws_feed.is_live() in the gate).
    if not ws_only and not _binance_reachable():
        log.error("[red]Cannot reach Binance API[/] - check network or VPN")
        return

    # ---- STEP 0.5: pre-warm the Market Map OUTSIDE the analysis burst ----
    # Its ~300 request-weight then ages out of the sliding 60s window while
    # the per-symbol analysis ramps up, instead of stacking right after it.
    # v5.4: also run the market cycle via the leader coins (BTC/ETH/SOL/XRP)
    # - trend/RSI/momentum read + market-wide verdict + the human-readable
    # classification file (data/market_groups.txt) refreshed every hour.
    # v5.14: skipped in degraded mode - the cached map stays authoritative
    # and no REST rebuild is poked while a ban is active.
    if not ws_only:
        try:
            from src.analysis.market_map import market_map
            market_map.get_map()
            mk = (market_map.run_market_cycle() or {}).get("market") or {}
            if mk:
                log.info(
                    f"[bold cyan]Market posture:[/] {mk.get('verdict')} "
                    f"({mk.get('score')}/100) - {mk.get('posture_ar')}"
                )
        except Exception as e:
            log.debug(f"Market map pre-warm skipped: {e}")

    cycle_start = time.time()

    # ---- STEP 1: manage open positions (SL/TP + partial + trailing) ----
    manage_open_positions()

    # ---- STEP 1.5: regime router (v5.13) - classify the market state and
    # log the resulting policy once per cycle (cached 15 min, no extra REST
    # weight: leaders come from the market cycle cache, F&G is a free API
    # cached 4h, volatility reads the WS candle cache).
    if settings.REGIME_ENABLED:
        try:
            from src.analysis.regime_router import regime_router
            regime_router.get_regime()
        except Exception as e:
            log.debug(f"Regime router skipped: {e}")

    # ---- STEP 2: market analysis ----
    recommendations = analyzer_analyze(ws_only=ws_only)

    # v5.12: the burst was aborted mid-way by a Binance 429/418 ban. The
    # last good recommendations snapshot is untouched; sending "no signals"
    # notifications or trading on the empty result would be wrong. The
    # 1-minute position watcher stays fully armed either way.
    try:
        from src.analysis.analyzer import analyzer
        aborted = getattr(analyzer, "last_run_aborted", False)
    except Exception:
        aborted = False
    if aborted:
        log.warning(
            "[yellow]Cycle aborted by rate ban[/] - no notifications sent, "
            "last recommendations snapshot kept"
        )
        return

    # ---- STEP 3: trailing + structural exits with FRESH signals ----
    if risk_manager.open_positions and recommendations:
        try:
            market_signals = _market_signals_from_file()
            manage_open_positions(market_signals, structural=True)
        except Exception as e:
            log.error(f"Post-analysis management failed: {e}")

    if not recommendations:
        log.info("[yellow]No strong signals in this cycle.[/]")
        log.info(
            f"[green]Cycle complete[/] in {time.time()-cycle_start:.1f}s "
            f"(open positions: {len(risk_manager.open_positions)}, "
            f"pending: {len(risk_manager.pending_entries)})"
        )
        return

    # ---- STEP 3.5: AI advisor - Arabic technical comments (v5.9, optional) ----
    # Adds rec["ai_comment"] via the user's CodeCraft API key. Optional:
    # disabled without a key, and any failure is silently skipped. The
    # enriched list is re-saved so the dashboard + WS see the comment too.
    try:
        from src.ai import ai_advisor
        if ai_advisor.enrich_recommendations(recommendations):
            snap = load_json(RECOMMENDATIONS_FILE, default={})
            if isinstance(snap, dict):
                snap["top_recommendations"] = to_json_safe(recommendations)
                save_json(snap, RECOMMENDATIONS_FILE)
    except Exception as e:
        log.warning(f"AI advisor skipped: {e}")

    # ---- STEP 4: log + notify ----
    file_logger.log_recommendations(recommendations)
    telegram_notifier.send_recommendations(recommendations)

    # ---- STEP 5: open new positions (veteran gates + pending entries) ----
    log.info(f"[cyan]Mode:[/] {settings.RUN_MODE}")
    opened = open_new_positions(recommendations)

    log.info(
        f"[green]Analysis cycle complete[/] in {time.time()-cycle_start:.1f}s - "
        f"opened {opened} new positions "
        f"(total open: {len(risk_manager.open_positions)}, "
        f"pending: {len(risk_manager.pending_entries)})"
    )


def analyzer_analyze(ws_only: bool = False):
    """Lazy import to avoid circulars (analyzer imports scorer chains).

    v5.14: `ws_only=True` runs the whole burst off the WS candle cache with
    order books skipped (zero REST weight) - used while a rate ban is active.
    """
    from src.analysis.analyzer import analyzer
    return analyzer.analyze_all(parallel=True, ws_only=ws_only)
