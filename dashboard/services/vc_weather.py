# dashboard/services/vc_weather.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional, Tuple

import pandas as pd
import requests
from django.conf import settings
from django.utils import timezone


@dataclass
class WeatherResult:
    ok: bool
    source: str
    df: pd.DataFrame
    error: Optional[str] = None


def _now_local() -> datetime:
    return timezone.localtime(timezone.now()).replace(minute=0, second=0, microsecond=0)


def _visual_crossing_hourly_df(data: dict) -> pd.DataFrame:
    rows = []
    for day in data.get("days", []) or []:
        for h in day.get("hours", []) or []:
            # datetime string like "2025-12-19T10:00:00"
            dt_str = h.get("datetime")  # "10:00:00" in some responses
            if "datetimeEpoch" in h:
                dt = datetime.fromtimestamp(h["datetimeEpoch"], tz=timezone.get_current_timezone())
            else:
                # fallback: day["datetime"] + hour
                base = day.get("datetime")
                if base and dt_str:
                    dt = datetime.fromisoformat(f"{base}T{dt_str}").replace(tzinfo=timezone.get_current_timezone())
                else:
                    continue

            rows.append(
                {
                    "ds": dt,
                    "irradiation": h.get("solarradiation"),  # W/m2
                    "air_temp": h.get("temp"),
                    # Visual Crossing metric API returns windspeed in km/h; store m/s internally.
                    "wind_speed": (h.get("windspeed") / 3.6) if h.get("windspeed") is not None else None,
                    "cloudcover": h.get("cloudcover"),
                    "humidity": h.get("humidity"),
                    "precip": h.get("precip"),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # нормализуем типы
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds").reset_index(drop=True)
    df["ds"] = df["ds"].dt.floor("h")  # pandas future-safe

    # numeric
    for c in ["irradiation", "air_temp", "wind_speed", "cloudcover", "humidity", "precip"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def _fetch_visual_crossing_timeline(lat: float, lon: float, start_date: date, end_date: date, *, source: str) -> WeatherResult:
    api_key = getattr(settings, "VISUAL_CROSSING_API_KEY", None)
    if not api_key:
        return WeatherResult(ok=False, source=source, df=pd.DataFrame(), error="VISUAL_CROSSING_API_KEY missing")

    # Visual Crossing timeline API (metric) works for both forecast and historical dates.
    url = (
        f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"
        f"{lat},{lon}/{start_date}/{end_date}"
    )
    params = {
        "unitGroup": "metric",
        "include": "hours",
        "key": api_key,
        "contentType": "json",
    }

    try:
        r = requests.get(url, params=params, timeout=45)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return WeatherResult(ok=False, source=source, df=pd.DataFrame(), error=str(e))

    df = _visual_crossing_hourly_df(data)
    if df.empty:
        return WeatherResult(ok=False, source=source, df=df, error="Empty VC response")
    return WeatherResult(ok=True, source=source, df=df)


def fetch_visual_crossing_hourly(lat: float, lon: float, days: int) -> WeatherResult:
    """
    Возвращает почасовой прогноз Visual Crossing на N дней вперёд в датафрейме:
    ds, irradiation, air_temp, wind_speed, cloudcover, humidity, precip
    """
    days = max(int(days), 1)
    # Forecast UI semantics: 1 day means tomorrow only, not today + tomorrow.
    start = _now_local() + timedelta(days=1)
    end = start + timedelta(days=days - 1)
    return _fetch_visual_crossing_timeline(lat, lon, start.date(), end.date(), source="visual_crossing")


def fetch_visual_crossing_hourly_range(lat: float, lon: float, start_date: date, end_date: date) -> WeatherResult:
    """
    Возвращает почасовые данные Visual Crossing за явный диапазон дат.
    Используется для постфактум-прогноза, когда локальной ветровой истории нет.
    """
    return _fetch_visual_crossing_timeline(lat, lon, start_date, end_date, source="visual_crossing_postfactum")

