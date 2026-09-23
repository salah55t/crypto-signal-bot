"""
Main bot runner (v5) - thin wrapper around the unified trading cycle.

v5: all logic lives in src/core/cycle.py (shared with the web dashboard so
both paths behave identically). This process runs:
  - the full analysis cycle every SCHEDULE_CRON (default */10 min)
  - a 1-minute real-time position watcher (SL/TP/partial/pending fills)
"""
import sys
from pathlib import Path

# Ensure project root on path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config.settings import settings
from src.core.scheduler import scheduler
from src.core.cycle import run_analysis_cycle, run_position_watch
from src.risk.manager import risk_manager
from src.utils.logger import log


def main():
    """Run once or schedule."""
    log.info("[bold green]Crypto Signal Bot starting up (v5 veteran engine)...[/]")
    log.info(f"[cyan]Mode:[/] {settings.RUN_MODE}")
    log.info(f"[cyan]Min confidence:[/] {settings.MIN_CONFIDENCE}% | "
             f"Min harmony:[/] {settings.MIN_HARMONY:.2f}")
    log.info(f"[cyan]Partial TP:[/] {settings.PARTIAL_TP_ENABLED} | "
             f"Chandelier:[/] {settings.CHANDELIER_ENABLED} "
             f"({settings.CHANDELIER_ATR_MULT}xATR) | "
             f"Structural exits:[/] {settings.STRUCTURAL_EXITS_ENABLED}")
    log.info(f"[cyan]Pending limit entries:[/] {settings.PENDING_ENTRIES_ENABLED}")
    log.info(f"[cyan]Timeframes:[/] {', '.join(settings.TIMEFRAMES)}")
    log.info(f"[cyan]Schedule cron:[/] '{settings.SCHEDULE_CRON}' + 1-min watcher")
    log.info(f"[cyan]Open positions:[/] {len(risk_manager.open_positions)} | "
             f"Pending: {len(risk_manager.pending_entries)}")

    if "--once" in sys.argv:
        # Single run
        run_analysis_cycle()
    else:
        # Initial run
        run_analysis_cycle()
        # Then schedule recurring: full cycle + fast watcher
        scheduler.add_job(run_analysis_cycle, job_id="analyze")
        scheduler.add_job(
            run_position_watch, cron="* * * * *", job_id="position_watch"
        )
        scheduler.start()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("[yellow]Bot stopped by user[/]")
    except Exception as e:
        log.exception(f"[red]Fatal error[/]: {e}")
