from __future__ import annotations

from django.core.management.base import BaseCommand

from dashboard.services.forecast_scheduler import run_scheduled_forecasts


class Command(BaseCommand):
    help = "Run scheduled forecasts configured in the portal."

    def handle(self, *args, **options):
        updated_rows = 0
        try:
            # Импортируем здесь, чтобы поломка модуля автоистории
            # не блокировала запуск расписанных прогнозов целиком.
            from dashboard.services.history_autofill import run_auto_history_updates

            updated_rows = run_auto_history_updates()
        except Exception as exc:
            self.stdout.write(self.style.WARNING(f"Auto history skipped due to error: {exc}"))

        forecast_count = run_scheduled_forecasts()
        self.stdout.write(self.style.SUCCESS(f"Auto history rows upserted: {updated_rows}"))
        self.stdout.write(self.style.SUCCESS(f"Scheduled forecasts executed: {forecast_count}"))
