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

DATA_DIR = Path("data")
CLOSED_TRADES_FILE = DATA_DIR / "closed_trades.json"
RECOMMENDATIONS_FILE = DATA_DIR / "recommendations.json"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def fetch_prices(symbols) -> Dict[str, float]:
    """v5: batched price fetch - ONE request (weight 2-4), not 80."""
    if not symbols:
        return {}
    try:
        return data_fetcher.get_batch_prices(symbols)
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


def _notify_closed(closed: list):
    for c in closed:
        if not telegram_notifier.enabled:
            continue
        if c.get("status") == "partial":
            telegram_notifier.send_alert(
                f"TP1 Partial 🎯 {c.get('symbol', '')}",
                f"Banked {c.get('fraction', 0)*100:.0f}% at {c.get('exit_price')}\n"
                f"PnL: ${c.get('pnl', 0):+.2f} ({c.get('pnl_pct', 0):+.2f}%)\n"
                f"Runner now targets TP2 with SL at break-even"
            )
            continue
        win_emoji = "✅" if c.get("pnl", 0) > 0 else "❌"
        telegram_notifier.send_alert(
            f"Position Closed {win_emoji}",
            f"Symbol: {c.get('symbol')}\n"
            f"Entry: {c.get('entry_price')}\n"
            f"Exit: {c.get('exit_price')}\n"
            f"PnL: ${c.get('pnl', 0):+.2f} ({c.get('pnl_pct', 0):+.2f}%)\n"
            f"Reason: {c.get('reason', '')}\n"
            f"Mode: {'PAPER' if c.get('paper', True) else 'LIVE'}"
        )


def _notify_updates(updates: list, tag: str = ""):
    for u in updates:
        if telegram_notifier.enabled:
            telegram_notifier.send_alert(
                f"Risk Update 🔧 {u.get('symbol', '')} {tag}".strip(),
                f"{u.get('reason', '')}\n"
                f"Old SL: {u.get('old_sl')} → New SL: {u.get('new_sl')}\n"
                f"Old TP: {u.get('old_tp')} → New TP: {u.get('new_tp')}"
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
    current_prices = fetch_prices(pos_symbols)
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
        prices = fetch_prices(symbols)
        if not prices:
            return

        # 1) pending limit fills (price returned to the entry zone)
        if risk_manager.pending_entries:
            filled = risk_manager.check_pending_fills(prices)
            for f in filled:
                if telegram_notifier.enabled:
                    pos = f.get("position", {})
                    telegram_notifier.send_alert(
                        f"Limit Entry Filled 🎯 {f.get('symbol', '')}",
                        f"Filled at: {f.get('fill_price')}\n"
                        f"Entry zone respected (no chasing)\n"
                        f"SL: {pos.get('stop_loss')} | TP: {pos.get('take_profit')}"
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
                        f"Pending Limit Entry ⏳ {symbol}",
                        f"Entry zone: {pending['zone_low']:.4f} - "
                        f"{pending['zone_high']:.4f}\n"
                        f"Current: {price:.4f} (waiting for pullback)\n"
                        f"SL: {rec.get('stop_loss')} | TP1: {rec.get('take_profit')} "
                        f"| TP2: {rec.get('take_profit_2')}\n"
                        f"Expires in {settings.PENDING_TTL_HOURS:.0f}h"
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
                    f"Position Opened 🚀 {symbol}",
                    f"Mode: {mode_tag}\n"
                    f"Entry: {pos.get('entry_price')} "
                    f"({pos.get('entry_type', 'market')})\n"
                    f"SL: {pos.get('stop_loss')}\n"
                    f"TP1: {pos.get('take_profit')} (banks 50%)\n"
                    f"TP2: {pos.get('take_profit_2')} (runner)\n"
                    f"Confidence: {rec.get('confidence', 0):.0f}% | "
                    f"Harmony: {rec.get('harmony', 0):.2f}"
                )
        else:
            log.warning(
                f"Position rejected: {result.get('reasons', result.get('reason'))}"
            )
    return opened


# ------------------------------------------------------------------
# FULL CYCLE
# ------------------------------------------------------------------
def rate_limit_gate() -> bool:
    """True when this cycle should SKIP because of active rate limiting.

    Distinguishes "rate-limited (transient - next cron tick retries)" from
    "network down" so the log stops showing a misleading network error when
    the limiter itself is protecting the shared IP (429/418 backoff).
    """
    from src.core.rate_limiter import rate_limiter
    remaining = rate_limiter.cooldown_remaining()
    if remaining > 0:
        log.warning(
            f"[yellow]Rate-limit cooldown active ({remaining:.0f}s left) - "
            f"skipping this cycle (429/418 or shared-IP pressure); "
            f"next cron tick retries automatically[/]"
        )
        return True
    return False


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
    """One full bot cycle: manage -> analyze -> manage(signals) -> notify -> open."""
    log.info("=" * 60)
    log.info("[bold cyan]STARTING ANALYSIS CYCLE (v5)[/]")
    log.info("=" * 60)

    # ---- STEP 0: rate-limit gate (429/418 or shared-IP pressure) ----
    if rate_limit_gate():
        return

    if not _binance_reachable():
        log.error("[red]Cannot reach Binance API[/] - check network or VPN")
        return

    # ---- STEP 0.5: pre-warm the Market Map OUTSIDE the analysis burst ----
    # Its ~300 request-weight then ages out of the sliding 60s window while
    # the per-symbol analysis ramps up, instead of stacking right after it.
    # v5.4: also run the market cycle via the leader coins (BTC/ETH/SOL/XRP)
    # - trend/RSI/momentum read + market-wide verdict + the human-readable
    # classification file (data/market_groups.txt) refreshed every hour.
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

    # ---- STEP 2: market analysis ----
    recommendations = analyzer_analyze()

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


def analyzer_analyze():
    """Lazy import to avoid circulars (analyzer imports scorer chains)."""
    from src.analysis.analyzer import analyzer
    return analyzer.analyze_all(parallel=True)
