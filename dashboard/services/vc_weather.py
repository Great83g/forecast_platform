# dashboard/services/vc_weather.py

import requests
import pandas as pd
import numpy as np
from datetime import datetime

# Можно вынести в settings, но для простоты оставим тут
VC_API_KEY = "WFZVPPR44XXZALVNSDDWDALPU"


def load_weather(lat: float, lon: float, hours_ahead: int = 72) -> pd.DataFrame:
    """
    Загружает прогноз Visual Crossing на N часов вперёд
    и возвращает DataFrame с колонками:
      ds, Irradiation, Air_Temp, PV_Temp, cloudcover, hour, hour_sin, hour_cos
    """
    url = (
        "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"
        f"{lat},{lon}/next{hours_ahead}hours"
        f"?unitGroup=metric&include=hours&key={VC_API_KEY}&contentType=json"
    )

    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()

    hours = []
    for day in data.get("days", []):
        day_date = day.get("datetime")
        for h in day.get("hours", []):
            # day_date = '2025-12-09', h['datetime'] = '06:00:00+06:00'
            time_str = h.get("datetime", "00:00:00")
            # обрежем TZ, если есть
            if "+" in time_str or "-" in time_str[2:]:
                time_str = time_str.split("+")[0].split("-")[0]
            dt_str = f"{day_date}T{time_str}"
            ts = datetime.fromisoformat(dt_str)

            hours.append(
                {
                    "ds": ts,
                    "Irradiation": float(h.get("solarradiation", 0)),
                    "Air_Temp": float(h.get("temp", 0)),
                    "PV_Temp": float(h.get("feelslike", 0)),
                    "cloudcover": float(h.get("cloudcover", 0)),
                }
            )

    df = pd.DataFrame(hours)
    if df.empty:
        return df

    df["ds"] = pd.to_datetime(df["ds"])
    df["hour"] = df["ds"].dt.hour

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)

    return df
