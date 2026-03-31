"""Daily scheduler — runs the pipeline once per day at a configured time."""

from __future__ import annotations

import schedule
import time

from src.config import Settings, get_settings
from src.logging_utils import get_logger, setup_logging

log = get_logger("scheduler")


def daily_job(settings: Settings) -> None:
    """Execute the full pipeline as a scheduled daily job."""
    log.info("Daily job triggered")
    try:
        from src.main import run_pipeline

        run_pipeline(settings)
    except Exception:
        log.exception("Daily job failed")


def start_scheduler(settings: Settings | None = None) -> None:
    """Start the blocking scheduler loop. Runs forever."""
    if settings is None:
        settings = get_settings()

    setup_logging(settings.log_level, settings.log_file)
    log.info("Scheduler starting — daily run at %s", settings.schedule_time)

    schedule.every().day.at(settings.schedule_time).do(daily_job, settings=settings)

    # Run immediately on first launch, then wait
    log.info("Running initial job now…")
    daily_job(settings)

    while True:
        schedule.run_pending()
        time.sleep(60)
