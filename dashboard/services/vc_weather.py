# dashboard/services/vc_weather.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
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


def fetch_visual_crossing_hourly(lat: float, lon: float, days: int) -> WeatherResult:
    """
    Возвращает почасовой прогноз Visual Crossing на N дней вперёд в датафрейме:
    ds, irradiation, air_temp, wind_speed, cloudcover, humidity, precip
    """
    api_key = getattr(settings, "VISUAL_CROSSING_API_KEY", None)
    if not api_key:
        return WeatherResult(ok=False, source="visual_crossing", df=pd.DataFrame(), error="VISUAL_CROSSING_API_KEY missing")

    start = _now_local()
    end = start + timedelta(days=days)

    # Visual Crossing timeline API (metric)
    url = (
        f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"
        f"{lat},{lon}/{start.date()}/{end.date()}"
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
        return WeatherResult(ok=False, source="visual_crossing", df=pd.DataFrame(), error=str(e))

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
                    "wind_speed": h.get("windspeed"),
                    "cloudcover": h.get("cloudcover"),
                    "humidity": h.get("humidity"),
                    "precip": h.get("precip"),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return WeatherResult(ok=False, source="visual_crossing", df=df, error="Empty VC response")

    # нормализуем типы
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds").reset_index(drop=True)
    df["ds"] = df["ds"].dt.floor("h")  # pandas future-safe

    # numeric
    for c in ["irradiation", "air_temp", "wind_speed", "cloudcover", "humidity", "precip"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return WeatherResult(ok=True, source="visual_crossing", df=df)

