from __future__ import annotations

import importlib
import logging
import time

from django.core.management.base import BaseCommand
from django.db import OperationalError
from django.utils import timezone

from dashboard.services.forecast_scheduler import run_scheduled_forecasts


logger = logging.getLogger(__name__)


def _run_auto_history_updates_safe(stdout, style) -> int:
    try:
        module = importlib.import_module("dashboard.services.history_autofill")
        run_auto_history_updates = getattr(module, "run_auto_history_updates")
        return int(run_auto_history_updates() or 0)
    except Exception as exc:
        logger.exception("Auto history update failed")
        stdout.write(
            style.WARNING(
                "Auto history skipped due to error. "
                f"Please check share path/migrations. Details: {exc}"
            )
        )
        return 0


class Command(BaseCommand):
    help = "Run scheduled forecasts configured in the portal."

    def handle(self, *args, **options):
        now_local = timezone.localtime()
        self.stdout.write(f"Scheduler tick at: {now_local:%Y-%m-%d %H:%M:%S %Z}")

        updated_rows = _run_auto_history_updates_safe(self.stdout, self.style)
        forecast_count = 0
        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                forecast_count = run_scheduled_forecasts(now=now_local)
                break
            except OperationalError as exc:
                is_locked = "locked" in str(exc).lower()
                if not is_locked or attempt >= max_attempts:
                    raise
                wait_seconds = attempt
                self.stdout.write(
                    self.style.WARNING(
                        f"Scheduler DB lock detected (attempt {attempt}/{max_attempts}). "
                        f"Retry in {wait_seconds}s."
                    )
                )
                time.sleep(wait_seconds)

        self.stdout.write(self.style.SUCCESS(f"Auto history rows upserted: {updated_rows}"))
        self.stdout.write(self.style.SUCCESS(f"Scheduled forecasts executed: {forecast_count}"))
