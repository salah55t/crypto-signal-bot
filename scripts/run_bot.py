"""
Main bot runner - performs one analysis cycle, manages open positions
(SL/TP checks + trailing stops), sends notifications, opens new paper/live
positions if applicable, and reschedules via scheduler.

v2: The old version NEVER checked open positions for SL/TP hits and NEVER
applied trailing stops (that logic existed only inside the web app). Running
`python scripts/run_bot.py` standalone meant positions stayed open forever
while Telegram claimed "auto-close every 1 minute". Fixed: every cycle now
starts with full position management before any new entry.
"""
import sys
import time
from pathlib import Path

# Ensure project root on path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import settings
from src.core.binance_client import binance_client
from src.core.scheduler import scheduler
from src.analysis.analyzer import analyzer
from src.risk.manager import risk_manager
from src.notifications import telegram_notifier, file_logger
from src.utils.logger import log
from src.utils.helpers import load_json, save_json, to_json_safe, now_utc

DATA_DIR = Path("data")
CLOSED_TRADES_FILE = DATA_DIR / "closed_trades.json"
RECOMMENDATIONS_FILE = DATA_DIR / "recommendations.json"


def _fetch_prices(symbols) -> dict:
    """Fetch current prices for a list of symbols. Returns {symbol: price}."""
    prices = {}
    try:
        tickers = binance_client.get_all_tickers()
        wanted = set(symbols)
        for t in tickers:
            if t["symbol"] in wanted:
                try:
                    prices[t["symbol"]] = float(t["lastPrice"])
                except (KeyError, ValueError, TypeError):
                    continue
    except Exception as e:
        log.error(f"Price fetch failed: {e}")
    return prices


def _save_closed_trades(closed: list):
    """Append closed trades to the closed-trades history file."""
    if not closed:
        return
    try:
        existing = load_json(CLOSED_TRADES_FILE, default=[])
        existing.extend(closed)
        save_json(to_json_safe(existing), CLOSED_TRADES_FILE)
    except Exception as e:
        log.error(f"Failed to save closed trades: {e}")


def _notify_closed(closed: list):
    """Send Telegram alerts for closed positions."""
    for c in closed:
        if not telegram_notifier.enabled:
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


def manage_open_positions(market_signals: dict = None) -> list:
    """
    STEP 1 of every cycle: manage existing positions.
    - Check SL/TP hits (paper AND live, same logic)
    - Apply trailing stop / break-even / TP-extension logic
    Returns list of closed positions.
    """
    market_signals = market_signals or {}

    if not risk_manager.open_positions:
        return []

    pos_symbols = list({p["symbol"] for p in risk_manager.open_positions})
    current_prices = _fetch_prices(pos_symbols)
    if not current_prices:
        log.warning("Could not fetch prices for open positions - skipping management")
        return []

    # 1) Close positions that hit SL/TP
    closed = risk_manager.check_open_positions(current_prices)
    for c in closed:
        log.info(
            f"[yellow]Position closed[/] {c['symbol']} - "
            f"PnL: ${c.get('pnl', 0):+.2f} ({c.get('pnl_pct', 0):+.2f}%) - {c.get('reason', '')}"
        )
    _save_closed_trades(closed)
    _notify_closed(closed)

    # 2) Apply trailing logic to remaining positions
    if risk_manager.open_positions:
        updates = risk_manager.apply_trailing_logic(current_prices, market_signals)
        for u in updates:
            log.info(f"[blue]Risk update[/] {u.get('reason', '')}")
            if telegram_notifier.enabled:
                telegram_notifier.send_alert(
                    f"Risk Update 🔧 {u.get('symbol', '')}",
                    f"{u.get('reason', '')}\n"
                    f"Old SL: {u.get('old_sl')} → New SL: {u.get('new_sl')}\n"
                    f"Old TP: {u.get('old_tp')} → New TP: {u.get('new_tp')}"
                )

    return closed


def run_analysis_cycle():
    """One full bot cycle: manage positions -> analyze -> notify -> open."""
    log.info("=" * 60)
    log.info("[bold cyan]STARTING ANALYSIS CYCLE[/]")
    log.info("=" * 60)

    # Verify Binance connectivity
    if not binance_client.ping():
        log.error("[red]Cannot reach Binance API[/] - check network or VPN")
        return

    cycle_start = time.time()

    # ---- STEP 1: Manage open positions (SL/TP + trailing) ----
    manage_open_positions()

    # ---- STEP 2: Market analysis ----
    recommendations = analyzer.analyze_all(parallel=True)

    # ---- STEP 3: trailing updates using fresh market signals ----
    if risk_manager.open_positions and recommendations:
        try:
            full_data = load_json(RECOMMENDATIONS_FILE, default={})
            market_signals = {
                r["symbol"]: r for r in full_data.get("all_results", [])
                if isinstance(r, dict) and "symbol" in r
            }
            pos_symbols = list({p["symbol"] for p in risk_manager.open_positions})
            current_prices = _fetch_prices(pos_symbols)
            risk_manager.apply_trailing_logic(current_prices, market_signals)
        except Exception as e:
            log.error(f"Post-analysis trailing update failed: {e}")

    if not recommendations:
        log.info("[yellow]No strong signals in this cycle.[/]")
        log.info(f"[green]Cycle complete[/] in {time.time()-cycle_start:.1f}s "
                 f"(open positions: {len(risk_manager.open_positions)})")
        return

    # ---- STEP 4: Log + notify ----
    file_logger.log_recommendations(recommendations)
    telegram_notifier.send_recommendations(recommendations)

    # ---- STEP 5: Open new positions (paper or live based on RUN_MODE) ----
    log.info(f"[cyan]Mode:[/] {settings.RUN_MODE}")
    open_symbols = {p["symbol"] for p in risk_manager.open_positions}
    opened = 0
    for rec in recommendations:
        if not risk_manager.can_open_position():
            log.warning("Max open positions or daily loss limit reached")
            break
        # Skip duplicates: never open two positions on the same symbol
        if settings.SKIP_DUPLICATE_SYMBOLS and rec["symbol"] in open_symbols:
            log.info(f"[yellow]Skip {rec['symbol']}[/] - position already open")
            continue
        result = risk_manager.open_position(rec)  # auto paper/live
        if result.get("status") == "opened":
            opened += 1
            open_symbols.add(rec["symbol"])
            mode_tag = "PAPER" if settings.RUN_MODE == "paper" else "LIVE"
            log.info(f"[green]{mode_tag} position opened for {rec['symbol']}[/]")
        else:
            log.warning(f"Position rejected: {result.get('reasons', result.get('reason'))}")

    log.info(
        f"[green]Analysis cycle complete[/] in {time.time()-cycle_start:.1f}s - "
        f"opened {opened} new positions (total open: {len(risk_manager.open_positions)})"
    )


def main():
    """Run once or schedule."""
    log.info("[bold green]Crypto Signal Bot starting up...[/]")
    log.info(f"[cyan]Mode:[/] {settings.RUN_MODE}")
    log.info(f"[cyan]Min confidence:[/] {settings.MIN_CONFIDENCE}%")
    log.info(f"[cyan]Min expected rise:[/] {settings.MIN_EXPECTED_RISE}%")
    log.info(f"[cyan]Timeframes:[/] {', '.join(settings.TIMEFRAMES)}")
    log.info(f"[cyan]Schedule cron:[/] '{settings.SCHEDULE_CRON}'")
    log.info(f"[cyan]Open positions:[/] {len(risk_manager.open_positions)}")

    if "--once" in sys.argv:
        # Single run
        run_analysis_cycle()
    else:
        # Initial run
        run_analysis_cycle()
        # Then schedule recurring
        scheduler.add_job(run_analysis_cycle, job_id="analyze")
        scheduler.start()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("[yellow]Bot stopped by user[/]")
    except Exception as e:
        log.exception(f"[red]Fatal error[/]: {e}")
