"""
Scheduler - runs the market analysis on a cron schedule.
"""
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from typing import Callable, Optional
from config.settings import settings
from src.utils.logger import log


class SignalScheduler:
    """Schedules periodic analysis runs."""

    def __init__(self):
        self.scheduler = BlockingScheduler(timezone="UTC")
        log.info(f"[cyan]Scheduler[/] initialized - cron: '{settings.SCHEDULE_CRON}'")

    def add_job(self, func: Callable, cron: Optional[str] = None,
                job_id: str = "analyze") -> None:
        """Add a scheduled job (v5: never overlapping instances)."""
        cron = cron or settings.SCHEDULE_CRON
        trigger = CronTrigger.from_crontab(cron)
        self.scheduler.add_job(
            func, trigger=trigger, id=job_id, name=job_id,
            misfire_grace_time=300,  # allow 5 min late
            max_instances=1,         # never run the same job twice at once
            coalesce=True,           # collapse missed runs into one
        )
        log.info(f"[green]Job added[/]: {job_id} (cron: '{cron}')")

    def start(self) -> None:
        """Start the scheduler (blocking)."""
        log.info("[green]Scheduler started[/] - press Ctrl+C to stop")
        try:
            self.scheduler.start()
        except (KeyboardInterrupt, SystemExit):
            log.info("[yellow]Scheduler stopped[/]")


# Singleton
scheduler = SignalScheduler()
