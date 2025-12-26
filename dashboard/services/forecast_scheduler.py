# dashboard/services/forecast_scheduler.py
from __future__ import annotations

from typing import Optional

from django.utils import timezone

from dashboard.models import ForecastSchedule
from dashboard.services.forecast_engine import run_forecast_for_station
from dashboard.services.forecast_reports import build_forecast_report, send_report_email


def _parse_providers(value: str) -> Optional[list[str]]:
    if not value:
        return None
    providers = [p.strip() for p in value.split(",") if p.strip()]
    return providers or None


def run_scheduled_forecasts(now: Optional[timezone.datetime] = None) -> int:
    current = now or timezone.localtime(timezone.now())
    today = current.date()
    run_count = 0

    for schedule in ForecastSchedule.objects.filter(enabled=True):
        if schedule.last_run_at and schedule.last_run_at.date() >= today:
            continue
        if current.time() < schedule.run_time:
            continue

        res = run_forecast_for_station(
            schedule.station_id,
            days=schedule.days,
            providers=_parse_providers(schedule.providers),
        )
        if res.get("ok"):
            report = build_forecast_report(
                station=schedule.station,
                days=schedule.days,
                weather_source=res.get("weather_source"),
                recipients=[schedule.emails],
            )
            send_report_email(report, [schedule.emails], schedule.station.name, schedule.days)

        schedule.last_run_at = current
        schedule.save(update_fields=["last_run_at"])
        run_count += 1

    return run_count
