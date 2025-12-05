# dashboard/services/forecast_engine.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import requests
from django.conf import settings
from django.db import transaction
from django.utils import timezone
from django.utils.timezone import make_aware
from neuralprophet import load as np_load
import xgboost as xgb

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


# ======================= АНСАМБЛЬ =======================

def ensemble(
    pred_np: np.ndarray,
    pred_xgb: np.ndarray,
    pred_heur: np.ndarray,
    min_value: float = 1.0,
    low_heur_threshold: float = 0.5,
    headroom: float = 1.5,
) -> np.ndarray:
    """
    Склеивает прогнозы, подстраховываясь эвристикой в «тёмные» часы и
    отбрасывая слишком оптимистичные модели.

    - Если эвристика < ``low_heur_threshold`` (почти нет солнца), NP/XGB
      принудительно обнуляются и итог совпадает с эвристикой.
    - Модельный прогноз учитывается только если его модуль ≥ ``min_value``
      и он не превышает эвристику больше чем в ``headroom`` раз.
    - Эвристика всегда участвует в усреднении как безопасный якорь.
    """

    pred_np_arr = np.asarray(pred_np, dtype=float)
    pred_xgb_arr = np.asarray(pred_xgb, dtype=float)
    pred_heur_arr = np.asarray(pred_heur, dtype=float)

    # В тёмное время полагаемся только на эвристику
    low_light_mask = pred_heur_arr < low_heur_threshold
    pred_np_arr = pred_np_arr.copy()
    pred_xgb_arr = pred_xgb_arr.copy()
    pred_np_arr[low_light_mask] = 0.0
    pred_xgb_arr[low_light_mask] = 0.0

    # Отбрасываем сильные всплески относительно базовой эвристики
    upper_bound = pred_heur_arr * headroom
    mask_np = (np.abs(pred_np_arr) >= min_value) & (pred_np_arr <= upper_bound)
    mask_xgb = (np.abs(pred_xgb_arr) >= min_value) & (pred_xgb_arr <= upper_bound)

    arr = np.vstack([pred_np_arr, pred_xgb_arr, pred_heur_arr])
    mask = np.vstack([mask_np, mask_xgb, np.ones_like(pred_heur_arr, dtype=bool)])

    denom = np.maximum(mask.sum(axis=0), 1)
    numer = (arr * mask).sum(axis=0)
    return numer / denom


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
    df["morning_peak_boost"] = ((df["hour"] == 6) & (df["Irradiation"] > 39)).astype(int)
    df["evening_penalty"] = ((df["hour"] == 19) & (df["Irradiation"] > 39)).astype(int)
    df["overdrive_flag"] = ((df["Irradiation"] > 950) & (df["Air_Temp"] > 30)).astype(int)
    df["midday_penalty"] = ((df["hour"] >= 12) & (df["hour"] <= 14)).astype(int)
    df["is_morning_active"] = ((df["hour"] == 6) & (df["Irradiation"] > 49)).astype(int)
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


def _ensure_1d(pred_like, length: int, name: str) -> np.ndarray:
    """Convert predictions to a flat float array and validate length.

    NeuralProphet и XGBoost иногда возвращают объекты с индексами, не совпадающими
    с текущим DataFrame. Явное приведение к numpy-формату фиксированной длины
    предотвращает «inhomogeneous shape» при последующих арифметических операциях.
    """

    try:
        series = pd.Series(pred_like)
    except Exception:
        # Если объект не приводится напрямую к Series, пробуем раскрыть его как
        # итерируемый список.
        series = pd.Series(list(pred_like))

    flat_values = []
    for idx, val in series.items():
        try:
            val_arr = np.asarray(val, dtype=float).reshape(-1)
        except Exception as e:
            # «setting an array element with a sequence» — пробуем раскрыть как
            # объектный массив и взять первый скалярный элемент, пригодный к float.
            try:
                obj_arr = np.array(val, dtype=object).ravel()
                first_scalar = next(float(x) for x in obj_arr if np.ndim(x) == 0)
                val_arr = np.asarray([first_scalar], dtype=float)
            except Exception:
                raise ValueError(
                    f"{name}: не удалось преобразовать элемент {idx} типа {type(val).__name__} -> float ({e})"
                )

        if val_arr.size == 0:
            raise ValueError(f"{name}: пустое значение в позиции {idx}")
        flat_values.append(float(val_arr[0]))

    arr = np.asarray(flat_values, dtype=float)

    if arr.shape[0] != length:
        raise ValueError(
            f"{name}: длина {arr.shape[0]} не совпадает с ожидаемой {length}"
        )
    return arr


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
    VC-погода → фичи → NeuralProphet(y) + XGB(PR) → калибровка → ансамбль.
    Логика адаптирована из «forecast_8p8mw_v11_fixed.py» с жёсткой
    фильтрацией ночи (Irr<20 или солнце<5° -> 0).
    """

    print(f"[FORECAST] station {station.pk}: запуск прогноза на {days} дней (NP+XGB+HEUR)")

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

    # === Модели ===
    np_path = MODEL_DIR / f"np_model_{station.pk}.np"
    np_meta_path = MODEL_DIR / f"np_model_{station.pk}.meta.json"
    xgb_path = MODEL_DIR / f"xgb_model_{station.pk}.json"
    xgb_meta_path = MODEL_DIR / f"xgb_model_{station.pk}.meta.json"

    np_model = None
    np_meta = {}
    if np_path.exists():
        try:
            np_model = np_load(str(np_path))
        except Exception as e:  # pragma: no cover - защитный лог
            print(f"[FORECAST] station {station.pk}: ошибка загрузки NP -> {e}")
    if np_meta_path.exists():
        try:
            np_meta = json.loads(np_meta_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[FORECAST] station {station.pk}: ошибка чтения NP meta -> {e}")

    xgb_model = None
    xgb_meta = {}
    if xgb_path.exists():
        try:
            xgb_model = xgb.XGBRegressor()
            xgb_model.load_model(str(xgb_path))
        except Exception as e:  # pragma: no cover - защитный лог
            print(f"[FORECAST] station {station.pk}: ошибка загрузки XGB -> {e}")
    if xgb_meta_path.exists():
        try:
            xgb_meta = json.loads(xgb_meta_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[FORECAST] station {station.pk}: ошибка чтения XGB meta -> {e}")

    b_exp = cal_factor(hist_df["y_mw"], expected_hist)
    print(f"[FORECAST] station {station.pk}: калибровка эвристики b_exp={b_exp:.3f}")

    xgb_features = xgb_meta.get(
        "X_cols",
        [
            "Irradiation",
            "Air_Temp",
            "PV_Temp",
            "hour",
            "month",
            "hour_sin",
            "month_sin",
            "sun_elev_deg",
            "low_sun_flag",
        ],
    )

    b_xgb = 1.0
    if xgb_model is not None:
        try:
            pred_hist_permw = np.clip(xgb_model.predict(hist_df[xgb_features]), 0, None)
            xgb_hist_mwh = pred_hist_permw * cap_mw
            b_xgb = cal_factor(hist_df["y_mw"], xgb_hist_mwh)
        except Exception as e:  # pragma: no cover - защитный лог
            print(f"[FORECAST] station {station.pk}: ошибка прогноза XGB на истории -> {e}")
            b_xgb = 1.0

    # калибровка NP усреднением с эвристикой
    b_np = (b_xgb + b_exp) / 2.0

    # === Погода на будущее ===
    df_hourly = fetch_weather_hours_for_station(station, days=days)
    if df_hourly.empty:
        print(f"[FORECAST] station {station.pk}: от VC пришёл пустой прогноз.")
        return 0

    df_hourly["Irradiation"] = df_hourly["Irradiation"].clip(lower=0.0)
    df_hourly.loc[df_hourly["Irradiation"] < 20.0, "Irradiation"] = 0.0

    df_hourly = _add_common_features(df_hourly, cap_mw=cap_mw)
    df_hourly = _add_sun_geometry(df_hourly, ds_col="ds", lat_deg=47.86)
    df_hourly = df_hourly.reset_index(drop=True)

    # expected по будущему (MW)
    budget_future = cap_mw * (df_hourly["Irradiation"] / 1000.0).clip(lower=eps)
    expected_future = (cap_mw * (df_hourly["Irradiation"] / 1000.0) * PR_BASE).clip(0, cap_mw * 0.95)
    cap_by_irr_future = budget_future * PR_CEIL_PER_HR

    # калиброванная эвристика
    df_hourly["Expected_MWh"] = expected_future
    df_hourly["Expected_MWh_cal"] = (expected_future * b_exp).clip(lower=0.0)

    # NP прогноз
    df_hourly["NeuralProphet_MWh_raw"] = 0.0
    df_hourly["NeuralProphet_MWh"] = 0.0
    df_hourly["NeuralProphet_MWh_cal"] = 0.0
    if np_model is not None:
        req_np = list(getattr(np_model, "config_regressors", {}).keys())
        missing = [r for r in req_np if r not in df_hourly.columns]
        if missing:
            print(f"[FORECAST] station {station.pk}: NP пропущены регрессоры {missing}")
        else:
            df_in_np = df_hourly[["ds"] + req_np].copy()
            df_in_np["y"] = np.nan
            fc_np = np_model.predict(df_in_np)
            np_yhat = _ensure_1d(fc_np["yhat1"], len(df_hourly), "NP yhat1")
            np_raw = (expected_future.to_numpy(dtype=float) + np.clip(np_yhat, 0, None)).clip(
                lower=0
            )
            cap_by_irr_arr = _ensure_1d(cap_by_irr_future, len(df_hourly), "cap_by_irr")
            phys_cap_arr = np.full(len(df_hourly), cap_mw, dtype=float)
            np_capped = np.minimum(np_raw, np.minimum(cap_by_irr_arr, phys_cap_arr))
            df_hourly["NeuralProphet_MWh"] = np_capped
            df_hourly["NeuralProphet_MWh_raw"] = np_raw
            df_hourly["NeuralProphet_MWh_cal"] = (np_capped * b_np).clip(lower=0.0)

    # XGB прогноз (PR -> MW)
    df_hourly["XGBoost_MWh_raw"] = 0.0
    df_hourly["XGBoost_MWh"] = 0.0
    df_hourly["XGBoost_MWh_cal"] = 0.0
    if xgb_model is not None:
        try:
            pred_permw = _ensure_1d(xgb_model.predict(df_hourly[xgb_features]), len(df_hourly), "XGB")
            pred_permw = np.clip(pred_permw, 0, None)
            xgb_raw = pred_permw * cap_mw
            cap_by_irr_arr = _ensure_1d(cap_by_irr_future, len(df_hourly), "cap_by_irr")
            phys_cap_arr = np.full(len(df_hourly), cap_mw, dtype=float)
            xgb_capped = np.minimum(xgb_raw, np.minimum(cap_by_irr_arr, phys_cap_arr))
            df_hourly["XGBoost_MWh"] = xgb_capped
            df_hourly["XGBoost_MWh_raw"] = xgb_raw
            df_hourly["XGBoost_MWh_cal"] = (xgb_capped * b_xgb).clip(lower=0.0)
        except Exception as e:  # pragma: no cover - защитный лог
            print(f"[FORECAST] station {station.pk}: ошибка прогноза XGB -> {e}")

    # Ансамбль по правилам из примера
    h = df_hourly["ds"].dt.hour
    irr = df_hourly["Irradiation"].fillna(0)
    low = (irr < 300) | (df_hourly["low_sun_flag"] == 1)

    w_np = np.where((h.between(6, 9)) & (irr > 60), 0.45, 0.30)
    w_xgb = np.where(h.between(6, 9), 0.35, 0.30)
    w_exp = 1.0 - (w_np + w_xgb)

    bright_midday = (h.between(11, 15)) & (irr > 700)
    w_exp = np.where(bright_midday, np.minimum(w_exp + 0.10, 0.85), w_exp)
    w_np = np.where(bright_midday, np.maximum(w_np - 0.05, 0.05), w_np)
    w_xgb = np.where(bright_midday, np.maximum(w_xgb - 0.05, 0.05), w_xgb)

    w_exp = np.where(low, np.minimum(w_exp + 0.20, 0.90), w_exp)
    w_np = np.where(low, np.maximum(w_np - 0.10, 0.05), w_np)
    w_xgb = np.where(low, np.maximum(w_xgb - 0.10, 0.05), w_xgb)

    w_np = w_np.astype(float)
    w_xgb = w_xgb.astype(float)
    w_exp = w_exp.astype(float)

    wsum = w_np + w_xgb + w_exp
    w_np, w_xgb, w_exp = w_np / wsum, w_xgb / wsum, w_exp / wsum

    df_hourly["Ensemble_MWh"] = (
        w_np * df_hourly["NeuralProphet_MWh_cal"]
        + w_xgb * df_hourly["XGBoost_MWh_cal"]
        + w_exp * df_hourly["Expected_MWh_cal"]
    )

    # Ночной фильтр
    mask_night = (df_hourly["Irradiation"] < 20) | (df_hourly["sun_elev_deg"] < 5)
    cols_zero = [
        "Expected_MWh",
        "Expected_MWh_cal",
        "NeuralProphet_MWh_raw",
        "NeuralProphet_MWh",
        "NeuralProphet_MWh_cal",
        "XGBoost_MWh_raw",
        "XGBoost_MWh",
        "XGBoost_MWh_cal",
        "Ensemble_MWh",
    ]
    df_hourly.loc[mask_night, cols_zero] = 0.0

    # === Сохранение в SolarForecast (MW -> кВт) ===
    timestamps = pd.to_datetime(df_hourly["ds"]).tolist()

    with transaction.atomic():
        SolarForecast.objects.filter(
            station=station,
            timestamp__in=timestamps,
        ).delete()

        objs = []
        for idx, (ds, v_exp, v_ens) in enumerate(
            zip(
                timestamps,
                df_hourly["Expected_MWh_cal"].values,
                df_hourly["Ensemble_MWh"].values,
                strict=True,
            )
        ):
            if timezone.is_naive(ds):
                ds = make_aware(ds, timezone.get_default_timezone())

            exp_kw = float(v_exp * 1000.0)
            ens_kw = float(v_ens * 1000.0)

            np_kw = float(df_hourly["NeuralProphet_MWh_cal"].iloc[idx] * 1000.0)
            xgb_kw = float(df_hourly["XGBoost_MWh_cal"].iloc[idx] * 1000.0)

            objs.append(
                SolarForecast(
                    station=station,
                    timestamp=ds,
                    pred_np=np_kw,
                    pred_xgb=xgb_kw,
                    pred_heur=exp_kw,
                    pred_final=ens_kw,
                )
            )

        SolarForecast.objects.bulk_create(objs)

    print(
        f"[FORECAST] station {station.pk}: прогноз записан в SolarForecast, строк: {len(timestamps)}, "
        f"cap_kw={cap_kw:.2f}"
    )
    return len(timestamps)

