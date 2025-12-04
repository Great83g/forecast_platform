import requests
import pandas as pd
from datetime import datetime, timedelta

VC_API_KEY = "WFZVPPR44XXZALVNSDDWDALPU"


def load_weather(lat, lon):
    """
    Загружает прогноз Visual Crossing на 72 часа вперёд
    и возвращает DataFrame с колонками:
    ds, Irradiation, Air_Temp, PV_Temp, hour_sin, hour_cos
    """
    url = (
        "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"
        f"{lat},{lon}"
        f"?unitGroup=metric&include=hours&key={VC_API_KEY}&contentType=json"
    )

    r = requests.get(url, timeout=15)
    r.raise_for_status()

    data = r.json()
    hours = []

    for day in data["days"]:
        for h in day["hours"]:
            ts = datetime.fromisoformat(h["datetime"][:-6])  # убираем TZ

            hours.append({
                "ds": ts,
                "Irradiation": float(h.get("solarradiation", 0)),
                "Air_Temp": float(h.get("temp", 0)),
                "PV_Temp": float(h.get("feelslike", 0)),   # временно используем feelslike
                "cloudcover": float(h.get("cloudcover", 0)),
            })

    df = pd.DataFrame(hours)

    # Циклические признаки
    df["hour"] = df["ds"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    return df

