"""
Main bot runner - performs one analysis cycle, sends notifications,
opens paper positions if applicable, and reschedules via scheduler.
"""
import sys
import os
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
from src.utils.helpers import fmt_price, fmt_pct


def run_analysis_cycle():
    """One full bot cycle: analyze -> log -> notify -> open paper positions."""
    log.info("=" * 60)
    log.info("[bold cyan]STARTING ANALYSIS CYCLE[/]")
    log.info("=" * 60)

    # Verify Binance connectivity
    if not binance_client.ping():
        log.error("[red]Cannot reach Binance API[/] - check network or VPN")
        return

    # Run analysis
    recommendations = analyzer.analyze_all(parallel=True, max_workers=5)

    if not recommendations:
        log.info("[yellow]No strong signals in this cycle.[/]")
        return

    # Log to file
    file_logger.log_recommendations(recommendations)

    # Send to Telegram
    telegram_notifier.send_recommendations(recommendations)

    # Open positions (paper or live based on RUN_MODE)
    log.info(f"[cyan]Mode:[/] {settings.RUN_MODE}")
    for rec in recommendations:
        if risk_manager.can_open_position():
            result = risk_manager.open_position(rec)  # auto paper/live
            if result.get("status") == "opened":
                mode_tag = "PAPER" if settings.RUN_MODE == "paper" else "LIVE"
                log.info(f"[green]{mode_tag} position opened for {rec['symbol']}[/]")
            else:
                log.warning(f"Position rejected: {result.get('reasons', result.get('reason'))}")
        else:
            log.warning("Max open positions or daily loss limit reached")
            break

    log.info("[green]Analysis cycle complete[/]")


def main():
    """Run once or schedule."""
    log.info("[bold green]Crypto Signal Bot starting up...[/]")
    log.info(f"[cyan]Mode:[/] {settings.RUN_MODE}")
    log.info(f"[cyan]Min confidence:[/] {settings.MIN_CONFIDENCE}%")
    log.info(f"[cyan]Min expected rise:[/] {settings.MIN_EXPECTED_RISE}%")
    log.info(f"[cyan]Timeframes:[/] {', '.join(settings.TIMEFRAMES)}")
    log.info(f"[cyan]Schedule cron:[/] '{settings.SCHEDULE_CRON}'")

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
