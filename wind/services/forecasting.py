from __future__ import annotations

from io import BytesIO

import pandas as pd
from django.core.files.base import ContentFile
from django.utils import timezone

from dashboard.models import ForecastReport
from dashboard.services.open_meteo import fetch_open_meteo_hourly
from dashboard.services.vc_weather import fetch_visual_crossing_hourly
from stations.models import Station

from wind.models import WindForecast
from wind.services.forecast_runs import dataframe_from_latest_wind_forecasts


def normalize_recipients(value: str) -> list[str]:
    if not value:
        return []
    return [p.strip() for p in value.replace(";", ",").split(",") if p.strip()]


def wind_power_kw_for_speed(station: Station, speed_ms: float | None) -> float:
    if speed_ms is None or pd.isna(speed_ms):
        return 0.0

    profile = getattr(station, "wind_profile", None)
    cut_in = float(getattr(profile, "cut_in_speed_ms", 3.0) or 3.0)
    rated = float(getattr(profile, "rated_speed_ms", 12.0) or 12.0)
    cut_out = float(getattr(profile, "cut_out_speed_ms", 25.0) or 25.0)
    capacity_kw = float(getattr(station, "capacity_ac_kw", 0.0) or 0.0)

    v = float(speed_ms)
    if v < cut_in or v >= cut_out or capacity_kw <= 0:
        return 0.0
    if v >= rated:
        return capacity_kw
    denominator = max(rated - cut_in, 0.1)
    normalized = max((v - cut_in) / denominator, 0.0)
    return capacity_kw * (normalized ** 3)


def fetch_weather_for_wind(station: Station, days: int, providers: list[str]) -> tuple[pd.DataFrame, str, list[str]]:
    errors = []
    results = []
    used_sources = []

    for provider in providers:
        if provider == "visual_crossing":
            wr = fetch_visual_crossing_hourly(station.latitude, station.longitude, days)
        elif provider == "open_meteo":
            wr = fetch_open_meteo_hourly(station.latitude, station.longitude, days, tz_name=station.timezone)
        else:
            continue

        if wr.ok and not wr.df.empty:
            df = wr.df.copy()
            if "wind_direction" in df.columns and "wind_direction_fc" not in df.columns:
                df["wind_direction_fc"] = pd.to_numeric(df.get("wind_direction"), errors="coerce")
            results.append(df)
            used_sources.append(wr.source)
        else:
            errors.append(f"{provider}: {wr.error or 'empty'}")

    if not results:
        return pd.DataFrame(), ",".join(used_sources), errors

    merged = pd.concat(results, ignore_index=True)
    numeric_cols = [
        c for c in ["irradiation", "air_temp", "wind_speed", "wind_direction_fc", "cloudcover", "humidity", "precip"] if c in merged.columns
    ]
    merged = merged.groupby("ds", as_index=False)[numeric_cols].mean(numeric_only=True)
    merged = merged.sort_values("ds").reset_index(drop=True)
    return merged, "+".join(sorted(set(used_sources))), errors


def build_wind_forecast_report(
    station: Station,
    scope: str,
    days: int,
    weather_source: str,
    recipients_raw: str = "",
    forecast_run=None,
) -> ForecastReport:
    if forecast_run is not None:
        qs = forecast_run.rows.filter(station=station, forecast_scope=scope).order_by("timestamp")
        df = pd.DataFrame(
            list(
                qs.values(
                    "timestamp",
                    "pred_heur",
                    "pred_final",
                    "wind_speed_fc",
                    "air_temp_fc",
                    "cloudcover_fc",
                    "humidity_fc",
                    "precip_fc",
                    "weather_source",
                )
            )
        )
    else:
        qs = WindForecast.objects.filter(station=station, forecast_scope=scope).select_related("forecast_run")
        df = dataframe_from_latest_wind_forecasts(qs)
    if not df.empty and "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        try:
            if getattr(df["timestamp"].dt, "tz", None) is not None:
                df["timestamp"] = df["timestamp"].dt.tz_convert(timezone.get_current_timezone())
                df["timestamp"] = df["timestamp"].dt.tz_localize(None)
            else:
                df["timestamp"] = df["timestamp"].dt.tz_localize(None)
        except Exception:
            pass
        for col in ["pred_heur", "pred_final"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce") / 1000.0
                df.rename(columns={col: f"{col}_mw"}, inplace=True)

    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="wind_forecast")
    out.seek(0)

    stamp = timezone.localtime(timezone.now()).strftime("%Y%m%d_%H%M%S")
    filename = f"wind_forecast_station_{station.pk}_{stamp}.xlsx"
    report = ForecastReport(
        station=station,
        days=days,
        weather_source=weather_source or "",
        recipients=", ".join(normalize_recipients(recipients_raw)),
    )
    report.file.save(filename, ContentFile(out.getvalue()), save=False)
    report.save()
    return report
