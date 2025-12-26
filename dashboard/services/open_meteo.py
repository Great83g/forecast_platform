# dashboard/services/open_meteo.py
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests
from django.conf import settings
from django.utils import timezone

from .vc_weather import WeatherResult


def _now_local() -> datetime:
    return timezone.localtime(timezone.now()).replace(minute=0, second=0, microsecond=0)


def _normalize_timezone(df: pd.DataFrame) -> pd.DataFrame:
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    try:
        if getattr(df["ds"].dt, "tz", None) is None:
            df["ds"] = df["ds"].dt.tz_localize(timezone.get_current_timezone())
        else:
            df["ds"] = df["ds"].dt.tz_convert(timezone.get_current_timezone())
    except Exception:
        pass
    return df


def _align_values(times: list, values: Optional[list]) -> list:
    if values is None:
        return [None] * len(times)
    if len(values) >= len(times):
        return values[: len(times)]
    return values + [None] * (len(times) - len(values))


def fetch_open_meteo_hourly(lat: float, lon: float, days: int) -> WeatherResult:
    """
    Возвращает почасовой прогноз Open-Meteo на N дней вперёд в датафрейме:
    ds, irradiation, air_temp, wind_speed, cloudcover, humidity, precip
    """
    start = _now_local()
    end = start + timedelta(days=days)

    base_url = getattr(settings, "OPEN_METEO_BASE_URL", "https://api.open-meteo.com/v1/forecast")
    timeout = getattr(settings, "OPEN_METEO_TIMEOUT", 45)
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(
            [
                "temperature_2m",
                "relativehumidity_2m",
                "precipitation",
                "cloudcover",
                "windspeed_10m",
                "shortwave_radiation",
            ]
        ),
        "start_date": start.date().isoformat(),
        "end_date": end.date().isoformat(),
        "timezone": "auto",
    }

    try:
        r = requests.get(base_url, params=params, timeout=timeout)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return WeatherResult(ok=False, source="open_meteo", df=pd.DataFrame(), error=str(e))

    hourly = data.get("hourly") or {}
    times = hourly.get("time") or []
    if not times:
        return WeatherResult(ok=False, source="open_meteo", df=pd.DataFrame(), error="Empty Open-Meteo response")

    df = pd.DataFrame(
        {
            "ds": times,
            "irradiation": _align_values(times, hourly.get("shortwave_radiation")),
            "air_temp": _align_values(times, hourly.get("temperature_2m")),
            "wind_speed": _align_values(times, hourly.get("windspeed_10m")),
            "cloudcover": _align_values(times, hourly.get("cloudcover")),
            "humidity": _align_values(times, hourly.get("relativehumidity_2m")),
            "precip": _align_values(times, hourly.get("precipitation")),
        }
    )

    df = _normalize_timezone(df)
    df = df.sort_values("ds").reset_index(drop=True)
    df["ds"] = df["ds"].dt.floor("h")

    for c in ["irradiation", "air_temp", "wind_speed", "cloudcover", "humidity", "precip"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return WeatherResult(ok=True, source="open_meteo", df=df)
