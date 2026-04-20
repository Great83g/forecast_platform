# dashboard/services/forecast_scheduler.py
from __future__ import annotations

import logging
import time
from datetime import timedelta
from typing import Optional

import pandas as pd
from django.conf import settings
from django.utils import timezone

from dashboard.models import ForecastSchedule
from dashboard.services.forecast_engine import run_forecast_for_station
from dashboard.services.forecast_reports import build_forecast_report, send_report_email
from stations.models import Station
from wind.services.forecasting import build_wind_forecast_report, fetch_weather_for_wind, wind_power_kw_for_speed


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


def _target_offsets_for_weekday_calendar(now_dt) -> list[int]:
    weekday = now_dt.weekday()
    if weekday == 4:  # Friday
        return [2, 3, 4]
    if weekday in {0, 1, 2, 3}:  # Mon-Thu
        return [2]
    return []


def _run_wind_scheduled_forecast(schedule: ForecastSchedule, current: timezone.datetime) -> bool:
    from wind.models import WindForecast

    station = schedule.station
    if station.latitude is None or station.longitude is None:
        logger.warning("Wind scheduled forecast skipped (no coords) station_id=%s", station.pk)
        return False

    providers = _parse_providers(schedule.providers) or ["visual_crossing", "open_meteo"]
    target_dates = None
    days = max(int(schedule.days or 1), 1)
    if (schedule.horizon_mode or "weekday_calendar") == "weekday_calendar":
        offsets = _target_offsets_for_weekday_calendar(current)
        if offsets:
            target_dates = {(current + timedelta(days=offset)).date() for offset in offsets}
            # Weather APIs могут отдавать горизонт "days" не включая весь последний календарный день.
            # Берём дополнительный день буфера, чтобы календарный режим стабильно отрабатывал.
            days = max(offsets) + 1

    weather_df, weather_source, errors = fetch_weather_for_wind(station, days, providers)
    if weather_df.empty:
        logger.warning(
            "Wind scheduled forecast failed station_id=%s schedule_id=%s errors=%s",
            schedule.station_id,
            schedule.pk,
            "; ".join(errors) or "empty weather",
        )
        return False

    if target_dates:
        base_weather_df = weather_df
        ds = pd.to_datetime(weather_df.get("ds"), errors="coerce")
        weather_df = weather_df.loc[ds.dt.date.isin(target_dates)].copy()
        if weather_df.empty:
            logger.warning(
                "Wind scheduled forecast has no rows for target dates, fallback to unfiltered horizon station_id=%s schedule_id=%s target_dates=%s",
                schedule.station_id,
                schedule.pk,
                sorted(target_dates),
            )
            weather_df = base_weather_df

    WindForecast.objects.filter(station=station, forecast_scope="main").delete()
    rows = []
    for _, row in weather_df.iterrows():
        speed = pd.to_numeric(row.get("wind_speed"), errors="coerce")
        pred = wind_power_kw_for_speed(station, speed)
        rows.append(
            WindForecast(
                station=station,
                forecast_scope="main",
                timestamp=row.get("ds").to_pydatetime() if hasattr(row.get("ds"), "to_pydatetime") else row.get("ds"),
                pred_heur=pred,
                pred_final=pred,
                weather_source=weather_source,
                air_temp_fc=float(row.get("air_temp")) if pd.notna(row.get("air_temp")) else None,
                wind_speed_fc=float(speed) if pd.notna(speed) else None,
                wind_direction_fc=float(row.get("wind_direction_fc")) if pd.notna(row.get("wind_direction_fc")) else None,
                cloudcover_fc=float(row.get("cloudcover")) if pd.notna(row.get("cloudcover")) else None,
                humidity_fc=float(row.get("humidity")) if pd.notna(row.get("humidity")) else None,
                precip_fc=float(row.get("precip")) if pd.notna(row.get("precip")) else None,
            )
        )
    WindForecast.objects.bulk_create(rows, batch_size=1000)

    recipients_configured = bool((schedule.emails or "").strip())
    if recipients_configured:
        report = build_wind_forecast_report(
            station=station,
            scope="main",
            days=days,
            weather_source=weather_source,
            recipients_raw=schedule.emails,
        )
        sent_ok = _send_report_with_retries(report, [schedule.emails], station.name, days)
        if not sent_ok:
            schedule.last_email_status = "Ошибка отправки email"
            schedule.save(update_fields=["last_email_status"])
            return False
        forecast_date = min(target_dates) if target_dates else (current + timedelta(days=1)).date()
        schedule.last_email_sent_at = current
        schedule.last_email_forecast_date = forecast_date
        schedule.last_email_status = f"Прогноз за {forecast_date:%d.%m.%Y} отправлен в {current:%H:%M}"
    return True

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
                if current < start_at:
                    continue

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

        if schedule.station.station_kind == Station.KIND_WIND:
            try:
                ok = _run_wind_scheduled_forecast(schedule, current)
            except Exception:
                logger.exception(
                    "Scheduled wind forecast crashed station_id=%s schedule_id=%s",
                    schedule.station_id,
                    schedule.pk,
                )
                continue
            if not ok:
                continue
            res = {"days": schedule.days}
        else:
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
        recipients_configured = bool((schedule.emails or "").strip())
        if recipients_configured:
            update_fields.extend(["last_email_sent_at", "last_email_forecast_date", "last_email_status"])
        schedule.save(update_fields=update_fields)
        run_count += 1

    return run_count
