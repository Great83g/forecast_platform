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


def _normalize_schedule_providers(value: str) -> tuple[Optional[list[str]], bool]:
    providers = _parse_providers(value) or []
    open_meteo_only = "open_meteo_only" in providers
    providers = [p for p in providers if p != "open_meteo_only"]
    if open_meteo_only:
        providers = ["open_meteo"]
    return (providers or None, open_meteo_only)


def run_scheduled_forecasts(now: Optional[timezone.datetime] = None, force: bool = False) -> int:
    current = now or timezone.localtime(timezone.now())
    today = current.date()
    run_count = 0

    for schedule in ForecastSchedule.objects.filter(enabled=True):
        if not force:
            if schedule.last_run_at and schedule.last_run_at.date() >= today:
                continue

            if schedule.start_at:
                start_at = timezone.localtime(schedule.start_at)
                if schedule.last_run_at is None:
                    if current < start_at:
                        continue
                else:
                    if current.time() < schedule.run_time:
                        continue
            else:
                if current.time() < schedule.run_time:
                    continue

        manual_dates = []
        if schedule.manual_snow_dates:
            for value in schedule.manual_snow_dates.split(","):
                value = value.strip()
                if not value:
                    continue
                try:
                    parsed = timezone.datetime.fromisoformat(value)
                    manual_dates.append(parsed.date())
                except ValueError:
                    continue

        providers, open_meteo_only = _normalize_schedule_providers(schedule.providers)

        res = run_forecast_for_station(
            schedule.station_id,
            days=schedule.days,
            providers=providers,
            manual_snow_enable=schedule.manual_snow_enable,
            manual_snow_factor=schedule.manual_snow_factor,
            manual_snow_dates=manual_dates,
            use_models=not open_meteo_only,
            horizon_mode=schedule.horizon_mode or "weekday_calendar",
            forecast_scope="main",
        )
        if res.get("ok"):
            effective_report_days = int(res.get("days") or schedule.days)
            report = build_forecast_report(
                station=schedule.station,
                days=effective_report_days,
                weather_source=res.get("weather_source"),
                recipients=[schedule.emails],
                forecast_scope="main",
                target_dates=res.get("target_dates") or [],
            )
            send_report_email(report, [schedule.emails], schedule.station.name, effective_report_days)

        schedule.last_run_at = current
        schedule.save(update_fields=["last_run_at"])
        run_count += 1

    return run_count
