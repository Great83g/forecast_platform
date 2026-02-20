from __future__ import annotations

import importlib

from django.core.management.base import BaseCommand

from dashboard.services.forecast_scheduler import run_scheduled_forecasts
from dashboard.services.history_autofill import run_auto_history_updates


def _run_auto_history_updates_safe(stdout, style) -> int:
    try:
        module = importlib.import_module("dashboard.services.history_autofill")
        run_auto_history_updates = getattr(module, "run_auto_history_updates")
        return int(run_auto_history_updates() or 0)
    except Exception as exc:
        stdout.write(style.WARNING(f"Auto history skipped due to error: {exc}"))
        return 0


class Command(BaseCommand):
    help = "Run scheduled forecasts configured in the portal."

    def handle(self, *args, **options):
<<<<<<< HEAD
        updated_rows = _run_auto_history_updates_safe(self.stdout, self.style)

=======
        updated_rows = run_auto_history_updates()
>>>>>>> origin/main
        forecast_count = run_scheduled_forecasts()
        self.stdout.write(self.style.SUCCESS(f"Auto history rows upserted: {updated_rows}"))
        self.stdout.write(self.style.SUCCESS(f"Scheduled forecasts executed: {forecast_count}"))
