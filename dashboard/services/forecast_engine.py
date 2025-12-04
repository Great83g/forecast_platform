# dashboard/services/forecast_engine.py

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import requests
from django.conf import settings
from django.db import transaction
from django.utils import timezone
from django.utils.timezone import make_aware

from solar.models import SolarRecord, SolarForecast

# Папка с моделями (пока не используем, но оставляем)
MODEL_DIR: Path = Path(settings.MODEL_DIR)

# API-ключ Visual Crossing — в settings.py:
# VISUAL_CROSSING_API_KEY = "WFZVPPR44XXZALVNSDDWDALPU"
VC_API_KEY: str = getattr(settings, "VISUAL_CROSSING_API_KEY", "")

# Часы дня (06–20)
DAY_START_H = 6
DAY_END_H = 20

# Параметры физики
PR_BASE = 0.90          # базовый PR для expected
PR_CEIL_PER_HR = 0.98   # потолок PR в час
ENSEMBLE_HEADROOM = 1.20   # запас наверх к expected (1.2 = +20%)


# ======================= УТИЛИТЫ ДЛЯ СТАНЦИИ =======================

def _get_station_coords(station) -> Tuple[float, float]:
    lat = getattr(station, "lat", None) or getattr(station, "latitude", None)
    lon = getattr(station, "lon", None) or getattr(station, "longitude", None)

    if lat is None or lon is None:
        raise ValueError(
            f"У станции {station.pk} нет координат (ожидались поля lat/lon или latitude/longitude)."
        )
    return float(lat), float(lon)


def _get_station_cap_kw(station, max_power_hist: float) -> float:
    cap_kw = getattr(station, "capacity_kw", None)
    if cap_kw is not None:
        try:
            v = float(cap_kw)
            if v > 0:
                return v
        except Exception:
            pass
    return float(max_power_hist or 0.0) or 10000.0  # запасной дефолт 10 МВт


# ======================= ФИЧИ =======================

def _add_sun_geometry(df: pd.DataFrame, ds_col: str = "ds", lat_deg: float = 47.86) -> pd.DataFrame:
    lat = np.deg2rad(lat_deg)
    doy = df[ds_col].dt.dayofyear
    hour = df[ds_col].dt.hour
    hour_angle = np.deg2rad((hour - 12) * 15)
    decl = np.deg2rad(23.44) * np.sin(2 * np.pi * (284 + doy) / 365)
    sin_elev = np.sin(lat) * np.sin(decl) + np.cos(lat) * np.cos(decl) * np.cos(hour_angle)
    df["sun_elev_deg"] = np.rad2deg(np.arcsin(np.clip(sin_elev, -1, 1)))
    df["low_sun_flag"] = (df["sun_elev_deg"] < 15).astype(int)
    return df


def _add_common_features(df: pd.DataFrame, cap_mw: float) -> pd.DataFrame:
    df["ds"] = pd.to_datetime(df["ds"])
    try:
        df["ds"] = df["ds"].dt.tz_localize(None)
    except TypeError:
        pass

    df["hour"] = df["ds"].dt.hour
    df["month"] = df["ds"].dt.month
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)

    df["is_clear"] = ((df["Irradiation"] > 200) & (df["Air_Temp"] > 0)).astype(int)
    df["PV_Temp"] = df["Air_Temp"] + np.maximum(df["Irradiation"] - 50, 0) / 1000.0 * 20.0

    # expected (MW)
    df["y_expected"] = cap_mw * (df["Irradiation"] / 1000.0) * PR_BASE
    df["y_expected"] = df["y_expected"].clip(upper=cap_mw * 0.95)
    df["y_expected_log"] = np.log1p(df["y_expected"] * 0.95)
    return df


# ======================= ПОГОДА ИЗ VISUAL CROSSING =======================

def fetch_weather_hours_for_station(station, days: int) -> pd.DataFrame:
    if not VC_API_KEY:
        raise RuntimeError("VISUAL_CROSSING_API_KEY не задан в settings.py")

    lat, lon = _get_station_coords(station)

    today = timezone.now().date()
    start_date = today + timezone.timedelta(days=1)
    end_date = start_date + timezone.timedelta(days=days - 1)

    url = (
        "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/"
        f"{lat},{lon}/{start_date:%Y-%m-%d}/{end_date:%Y-%m-%d}"
        f"?unitGroup=metric&include=hours&key={VC_API_KEY}&contentType=json"
    )

    print(f"[FORECAST] station {station.pk}: тянем Visual Crossing...\n  {url}")
    resp = requests.get(url, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"VC error: {resp.status_code} {resp.text}")

    data = resp.json()
    rows = []
    for d in data["days"]:
        for h in d["hours"]:
            h["datetime_full"] = f"{d['datetime']} {h['datetime']}"
            rows.append(h)

    dfw = pd.DataFrame(rows)
    dfw["datetime"] = pd.to_datetime(dfw["datetime_full"], errors="coerce")
    dfw.drop(columns=["datetime_full"], inplace=True)

    dfw = dfw.rename(
        columns={
            "solarradiation": "Irradiation",
            "temp": "Air_Temp",
            "windspeed": "Wind_Speed",
            "conditions": "Weather",
        }
    )

    dfw = dfw[(dfw["datetime"].dt.hour >= DAY_START_H) & (dfw["datetime"].dt.hour <= DAY_END_H)]

    keep = [
        "datetime",
        "Irradiation",
        "Air_Temp",
        "Wind_Speed",
        "cloudcover",
        "humidity",
        "precip",
        "Weather",
    ]
    dfw = dfw[[c for c in keep if c in dfw.columns]]

    dfw["ds"] = dfw["datetime"].dt.floor("H")
    dfh = dfw.groupby("ds", as_index=False).agg(
        {
            "Irradiation": "mean",
            "Air_Temp": "mean",
            "Wind_Speed": "mean",
            "cloudcover": "mean",
            "humidity": "mean",
            "precip": "mean",
        }
    )

    # базовая оценка PV_Temp
    dfh["PV_Temp"] = dfh["Air_Temp"] + np.maximum(dfh["Irradiation"] - 50, 0) / 1000.0 * 20.0

    print(f"[FORECAST] station {station.pk}: прогноз погоды готов, строк: {len(dfh)}")
    return dfh


# ======================= КАЛИБРОВКА =======================

def cal_factor(y_true: pd.Series, y_pred_like: pd.Series, lo=0.6, hi=1.0) -> float:
    s_true = float(np.nansum(y_true))
    s_pred = float(np.nansum(y_pred_like))
    if s_pred <= 1e-6:
        return 1.0
    return float(np.clip(s_true / s_pred, lo, hi))


def _load_history_df(station) -> pd.DataFrame:
    qs = SolarRecord.objects.filter(station=station).order_by("timestamp")
    if not qs.exists():
        return pd.DataFrame()

    rows = list(
        qs.values(
            "timestamp",
            "irradiation",
            "air_temp",
            "pv_temp",
            "power_kw",
        )
    )
    df = pd.DataFrame(rows)
    df.rename(
        columns={
            "timestamp": "ds",
            "irradiation": "Irradiation",
            "air_temp": "Air_Temp",
            "pv_temp": "PV_Temp",
            "power_kw": "Power_KW",
        },
        inplace=True,
    )
    df["ds"] = pd.to_datetime(df["ds"])
    try:
        df["ds"] = df["ds"].dt.tz_localize(None)
    except TypeError:
        pass

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["Power_KW", "Irradiation", "Air_Temp", "PV_Temp"])
    return df


# ======================= ОСНОВНОЙ ЗАПУСК ПРОГНОЗА (ЭВРИСТИКА) =======================

def run_forecast_for_station(station, days: int = 3) -> int:
    """
    Упрощённый, но стабильный вариант:
      - VC-погода → expected (эвристика) в MW
      - калибровка по истории станции
      - клипы по физике
      - запись в SolarForecast в кВт

    NP и XGB сейчас НЕ используются (pred_np=0, pred_xgb=0),
    итоговый прогноз = калиброванная эвристика.
    """

    print(f"[FORECAST] station {station.pk}: запуск прогноза на {days} дней (HEUR-ONLY)")

    # === История для мощности/калибровки ===
    hist_df = _load_history_df(station)
    if hist_df.empty:
        print(f"[FORECAST] station {station.pk}: нет истории для калибровки.")
        return 0

    max_power_hist = float(hist_df["Power_KW"].max() or 0.0)
    if max_power_hist <= 0.0:
        max_power_hist = 10000.0  # запасной дефолт, 10 МВт в кВт

    cap_kw = _get_station_cap_kw(station, max_power_hist)
    cap_mw = cap_kw / 1000.0
    print(f"[FORECAST] station {station.pk}: cap_kw≈{cap_kw:.1f}, cap_mw≈{cap_mw:.3f}")

    # История в MW + фичи
    hist_df["y_mw"] = hist_df["Power_KW"] / 1000.0
    hist_df = _add_common_features(hist_df, cap_mw=cap_mw)
    hist_df = _add_sun_geometry(hist_df, ds_col="ds", lat_deg=47.86)

    eps = 1e-6
    budget_hist = cap_mw * (hist_df["Irradiation"] / 1000.0).clip(lower=eps)
    expected_hist = (cap_mw * (hist_df["Irradiation"] / 1000.0) * PR_BASE).clip(0, cap_mw * 0.95)

    b_exp = cal_factor(hist_df["y_mw"], expected_hist)
    print(f"[FORECAST] station {station.pk}: калибровка эвристики b_exp={b_exp:.3f}")

    # === Погода на будущее ===
    df_hourly = fetch_weather_hours_for_station(station, days=days)
    if df_hourly.empty:
        print(f"[FORECAST] station {station.pk}: от VC пришёл пустой прогноз.")
        return 0

    df_hourly["Irradiation"] = df_hourly["Irradiation"].clip(lower=0.0)
    df_hourly.loc[df_hourly["Irradiation"] < 20.0, "Irradiation"] = 0.0

    df_hourly = _add_common_features(df_hourly, cap_mw=cap_mw)
    df_hourly = _add_sun_geometry(df_hourly, ds_col="ds", lat_deg=47.86)

    # expected по будущему (MW)
    budget_future = cap_mw * (df_hourly["Irradiation"] / 1000.0).clip(lower=eps)
    expected_future = (cap_mw * (df_hourly["Irradiation"] / 1000.0) * PR_BASE).clip(0, cap_mw * 0.95)

    # калиброванная эвристика
    df_hourly["Expected_MWh_cal"] = (expected_future * b_exp).clip(lower=0.0)

    # клип по физике
    cap_by_irr_future = budget_future * PR_CEIL_PER_HR
    lim_after = np.minimum.reduce(
        [
            cap_by_irr_future.values,
            np.full_like(cap_by_irr_future.values, cap_mw),
            (df_hourly["Expected_MWh_cal"].values * ENSEMBLE_HEADROOM),
        ]
    )
    df_hourly["Ensemble_MWh"] = np.minimum(df_hourly["Expected_MWh_cal"].values, lim_after)

    # Ночной фильтр
    mask_night = (df_hourly["Irradiation"] < 20) | (df_hourly["sun_elev_deg"] < 5)
    df_hourly.loc[mask_night, ["Expected_MWh_cal", "Ensemble_MWh"]] = 0.0

    # === Сохранение в SolarForecast (MW -> кВт) ===
    timestamps = pd.to_datetime(df_hourly["ds"]).tolist()

    with transaction.atomic():
        SolarForecast.objects.filter(
            station=station,
            timestamp__in=timestamps,
        ).delete()

        objs = []
        for ds, v_exp, v_ens in zip(
            timestamps,
            df_hourly["Expected_MWh_cal"].values,
            df_hourly["Ensemble_MWh"].values,
            strict=True,
        ):
            if timezone.is_naive(ds):
                ds = make_aware(ds, timezone.get_default_timezone())

            exp_kw = float(v_exp * 1000.0)
            ens_kw = float(v_ens * 1000.0)

            objs.append(
                SolarForecast(
                    station=station,
                    timestamp=ds,
                    pred_np=0.0,          # временно не используем
                    pred_xgb=0.0,         # временно не используем
                    pred_heur=exp_kw,     # эвристика (кВт)
                    pred_final=ens_kw,    # итог = эвристика после клипов
                )
            )

        SolarForecast.objects.bulk_create(objs)

    print(
        f"[FORECAST] station {station.pk}: прогноз записан в SolarForecast, строк: {len(timestamps)}, "
        f"cap_kw={cap_kw:.2f}"
    )
    return len(timestamps)

