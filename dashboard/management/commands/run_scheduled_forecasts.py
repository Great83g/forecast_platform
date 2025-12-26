from __future__ import annotations

from django.core.management.base import BaseCommand

from dashboard.services.forecast_scheduler import run_scheduled_forecasts


class Command(BaseCommand):
    help = "Run scheduled forecasts configured in the portal."

    def handle(self, *args, **options):
        count = run_scheduled_forecasts()
        self.stdout.write(self.style.SUCCESS(f"Scheduled forecasts executed: {count}"))
