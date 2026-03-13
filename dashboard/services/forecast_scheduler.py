# dashboard/services/forecast_scheduler.py
from __future__ import annotations

import logging
import time
from datetime import timedelta
from typing import Optional

from django.conf import settings
from django.utils import timezone

from dashboard.models import ForecastSchedule
from dashboard.services.forecast_engine import run_forecast_for_station
from dashboard.services.forecast_reports import build_forecast_report, send_report_email


logger = logging.getLogger(__name__)
LATE_RUN_WARN_MINUTES = 15


def _parse_providers(value: str) -> Optional[list[str]]:
    if not value:
        return None
    providers = [p.strip() for p in value.split(",") if p.strip()]
    return providers or None


def _normalize_schedule_providers(value: str) -> tuple[Optional[list[str]], bool]:
    providers = _parse_providers(value) or []
    open_meteo_only = "open_meteo_only" in providers
    visual_crossing_only = "visual_crossing_only" in providers
    providers = [p for p in providers if p not in {"open_meteo_only", "visual_crossing_only"}]
    heuristic_only = open_meteo_only or visual_crossing_only
    if heuristic_only and not providers:
        if open_meteo_only and visual_crossing_only:
            providers = ["visual_crossing", "open_meteo"]
        elif visual_crossing_only:
            providers = ["visual_crossing"]
        else:
            providers = ["open_meteo"]
    return (providers or None, heuristic_only)




def _send_report_with_retries(report, recipients, station_name: str, days: int) -> bool:
    max_attempts = max(1, int(getattr(settings, "FORECAST_EMAIL_MAX_ATTEMPTS", 3)))
    retry_delay_seconds = max(0, int(getattr(settings, "FORECAST_EMAIL_RETRY_DELAY_SECONDS", 15)))

    for attempt in range(1, max_attempts + 1):
        if send_report_email(report, recipients, station_name, days):
            return True

        logger.warning(
            "Scheduled report email attempt failed station=%s attempt=%s/%s",
            station_name,
            attempt,
            max_attempts,
        )
        if attempt < max_attempts and retry_delay_seconds:
            time.sleep(retry_delay_seconds)

    return False



def _resolve_report_forecast_date(result: dict, current: timezone.datetime):
    target_dates = result.get("target_dates") or []
    for value in target_dates:
        try:
            return timezone.datetime.fromisoformat(str(value)).date()
        except (TypeError, ValueError):
            continue
    return (current + timedelta(days=1)).date()

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

        scheduled_dt = timezone.datetime.combine(today, schedule.run_time, tzinfo=current.tzinfo)
        delay_minutes = max(0, int((current - scheduled_dt).total_seconds() // 60))
        if delay_minutes >= LATE_RUN_WARN_MINUTES:
            logger.warning(
                "Scheduled forecast late station_id=%s schedule_id=%s delay_minutes=%s now=%s run_time=%s",
                schedule.station_id,
                schedule.pk,
                delay_minutes,
                current.strftime("%Y-%m-%d %H:%M:%S%z"),
                schedule.run_time,
            )

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

        try:
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
        except Exception:
            logger.exception(
                "Scheduled forecast crashed station_id=%s schedule_id=%s",
                schedule.station_id,
                schedule.pk,
            )
            continue

        if not res.get("ok"):
            logger.warning(
                "Scheduled forecast failed station_id=%s schedule_id=%s result=%s",
                schedule.station_id,
                schedule.pk,
                res,
            )
            continue

        effective_report_days = int(res.get("days") or schedule.days)
        try:
            report = build_forecast_report(
                station=schedule.station,
                days=effective_report_days,
                weather_source=res.get("weather_source"),
                recipients=[schedule.emails],
                forecast_scope="main",
                target_dates=res.get("target_dates") or [],
            )
        except Exception:
            logger.exception(
                "Scheduled report build failed station_id=%s schedule_id=%s",
                schedule.station_id,
                schedule.pk,
            )
            continue

        recipients_configured = bool((schedule.emails or "").strip())
        if recipients_configured:
            sent_ok = _send_report_with_retries(report, [schedule.emails], schedule.station.name, effective_report_days)
            if not sent_ok:
                schedule.last_email_status = "Ошибка отправки email"
                schedule.save(update_fields=["last_email_status"])
                logger.warning(
                    "Scheduled report email failed station_id=%s schedule_id=%s recipients=%s",
                    schedule.station_id,
                    schedule.pk,
                    schedule.emails,
                )
                continue

            forecast_date = _resolve_report_forecast_date(res, current)
            schedule.last_email_sent_at = current
            schedule.last_email_forecast_date = forecast_date
            schedule.last_email_status = f"Прогноз за {forecast_date:%d.%m.%Y} отправлен в {current:%H:%M}"

        schedule.last_run_at = current
        update_fields = ["last_run_at"]
        if recipients_configured:
            update_fields.extend(["last_email_sent_at", "last_email_forecast_date", "last_email_status"])
        schedule.save(update_fields=update_fields)
        run_count += 1

    return run_count
