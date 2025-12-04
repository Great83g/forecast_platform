# dashboard/services/forecast_engine.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import requests
import xgboost as xgb
from django.conf import settings
from django.db import transaction
from django.utils import timezone
from django.utils.timezone import make_aware
from neuralprophet import NeuralProphet
from neuralprophet import load as np_load
from neuralprophet import save as np_save

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
TRAIN_IF_MISSING = True

XGB_FEATURES = [
    "Irradiation",
    "Air_Temp",
    "PV_Temp",
    "hour",
    "month",
    "hour_sin",
    "month_sin",
    "sun_elev_deg",
    "low_sun_flag",
]

NP_REGRESSORS = [
    "Irradiation",
    "Air_Temp",
    "PV_Temp",
    "hour_sin",
    "month_sin",
    "is_clear",
    "y_expected_log",
    "morning_peak_boost",
    "evening_penalty",
    "overdrive_flag",
    "midday_penalty",
    "is_morning_active",
    "sun_elev_deg",
    "low_sun_flag",
]


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
    """
    Возвращаем установленную мощность станции в кВт.

    - В модели Station поле называется ``capacity_mw`` (в МВт) — используем его в
      приоритете и переводим в кВт.
    - Если вдруг в объекте есть ``capacity_kw`` (например, из другого бэкэнда),
      тоже учитываем.
    - Фоллбек — максимум по истории, затем безопасный дефолт 10 МВт.
    """

    # 1) capacity_mw -> kW
    cap_mw = getattr(station, "capacity_mw", None)
    if cap_mw is not None:
        try:
            v_mw = float(cap_mw)
            if v_mw > 0:
                cap_kw_candidate = v_mw * 1000.0

                # Если значение в поле «MW» оказалось завышено на порядки относительно
                # наблюдаемой мощности, трактуем его как кВт (частая ошибка ввода).
                if max_power_hist > 0 and cap_kw_candidate > max_power_hist * 100:
                    return v_mw

                return cap_kw_candidate
        except Exception:
            pass

    # 2) legacy capacity_kw (если есть)
    cap_kw = getattr(station, "capacity_kw", None)
    if cap_kw is not None:
        try:
            v_kw = float(cap_kw)
            if v_kw > 0:
                return v_kw
        except Exception:
            pass

    # 3) максимум из истории или дефолт 10 МВт
    return float(max_power_hist or 0.0) or 10000.0


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


# ======================= ТРЕНИРОВКА МОДЕЛЕЙ =======================

def _train_np_residual(station, hist_df: pd.DataFrame, cap_mw: float) -> None:
    hist_df = hist_df.copy()
    hist_df["y_mw"] = hist_df["Power_KW"] / 1000.0
    hist_df = hist_df[hist_df["y_mw"] >= 0].copy()
    hist_df = _add_common_features(hist_df, cap_mw=cap_mw)
    hist_df = _add_sun_geometry(hist_df, ds_col="ds", lat_deg=47.86)

    hist_df["y_residual"] = hist_df["y_mw"] - hist_df["y_expected"]

    hist_df["dup_weight"] = 1
    hist_df.loc[hist_df["is_morning_active"] == 1, "dup_weight"] = 8
    df_dup = hist_df.loc[hist_df.index.repeat(hist_df["dup_weight"])].copy()
    df_dup["dup_index"] = df_dup.groupby("ds").cumcount()
    df_dup["ds"] = df_dup["ds"] + pd.to_timedelta(df_dup["dup_index"], unit="m")
    df_train = df_dup.drop(columns=["dup_index", "dup_weight"]).copy()

    cols = ["ds", "y_residual"] + NP_REGRESSORS
    df_train = df_train[cols].dropna().rename(columns={"y_residual": "y"})

    model = NeuralProphet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=True,
        seasonality_mode="additive",
        learning_rate=0.2,
        epochs=400,
        batch_size=64,
        loss_func="MSE",
        n_lags=0,
        n_forecasts=1,
    )
    for col in NP_REGRESSORS:
        model.add_future_regressor(col, normalize="minmax")

    model.fit(df_train, freq="h")

    path_model = MODEL_DIR / f"np_model_{station.pk}.np"
    path_meta = MODEL_DIR / f"np_model_{station.pk}.meta.json"
    np_save(model, str(path_model))
    path_meta.write_text(
        json.dumps(
            {
                "cap_mw_train": cap_mw,
                "pr_base": PR_BASE,
                "features_reg": NP_REGRESSORS,
                "note": "NeuralProphet residual: y_mw - expected(cap_mw)",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _train_xgb_permw(station, hist_df: pd.DataFrame, cap_mw: float) -> None:
    hist_df = hist_df.copy()
    hist_df["y_mw"] = hist_df["Power_KW"] / 1000.0
    hist_df = hist_df[hist_df["y_mw"] >= 0].copy()

    hist_df = _add_common_features(hist_df, cap_mw=cap_mw)
    hist_df = _add_sun_geometry(hist_df, ds_col="ds", lat_deg=47.86)

    df_train = hist_df.dropna(subset=XGB_FEATURES + ["y_mw"]).copy()
    y_permw = df_train["y_mw"] / cap_mw

    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )
    model.fit(df_train[XGB_FEATURES], y_permw)

    path_model = MODEL_DIR / f"xgb_model_{station.pk}.json"
    path_meta = MODEL_DIR / f"xgb_model_{station.pk}.meta.json"

    model.save_model(str(path_model))
    path_meta.write_text(
        json.dumps(
            {
                "cap_mw_train": cap_mw,
                "X_cols": XGB_FEATURES,
                "note": "XGB trained on y_per_mw",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _ensure_models(station, hist_df: pd.DataFrame, cap_mw: float) -> None:
    if not TRAIN_IF_MISSING:
        return

    if hist_df.empty:
        return

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    np_path = MODEL_DIR / f"np_model_{station.pk}.np"
    np_meta = MODEL_DIR / f"np_model_{station.pk}.meta.json"
    xgb_path = MODEL_DIR / f"xgb_model_{station.pk}.json"
    xgb_meta = MODEL_DIR / f"xgb_model_{station.pk}.meta.json"

    if not np_path.exists() or not np_meta.exists():
        _train_np_residual(station, hist_df, cap_mw)

    if not xgb_path.exists() or not xgb_meta.exists():
        _train_xgb_permw(station, hist_df, cap_mw)


# ======================= ОСНОВНОЙ ЗАПУСК ПРОГНОЗА (ЭВРИСТИКА) =======================

def run_forecast_for_station(station, days: int = 3) -> int:
    """
    Объединённый прогноз для станции:
      - VC-погода → expected (эвристика) в MW
      - автоматическая тренировка XGB per-MW и NeuralProphet residual при отсутствии моделей
      - калибровка моделей и эвристики по истории станции
      - динамический ансамбль NP/XGB/expected с клипами по физике и ночным фильтром
      - запись в SolarForecast в кВт
    """

    print(f"[FORECAST] station {station.pk}: запуск прогноза на {days} дней (ensemble)")

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

    _ensure_models(station, hist_df, cap_mw)

    # История в MW + фичи
    hist_df["y_mw"] = hist_df["Power_KW"] / 1000.0
    hist_df = _add_common_features(hist_df, cap_mw=cap_mw)
    hist_df = _add_sun_geometry(hist_df, ds_col="ds", lat_deg=47.86)

    eps = 1e-6
    budget_hist = cap_mw * (hist_df["Irradiation"] / 1000.0).clip(lower=eps)
    expected_hist = (cap_mw * (hist_df["Irradiation"] / 1000.0) * PR_BASE).clip(0, cap_mw * 0.95)

    b_exp = cal_factor(hist_df["y_mw"], expected_hist)
    print(f"[FORECAST] station {station.pk}: калибровка эвристики b_exp={b_exp:.3f}")

    b_xgb = 1.0
    b_np = b_exp

    # Калибровка XGB по истории
    xgb_path = MODEL_DIR / f"xgb_model_{station.pk}.json"
    xgb_meta_path = MODEL_DIR / f"xgb_model_{station.pk}.meta.json"
    if xgb_path.exists():
        try:
            xgb_model = xgb.XGBRegressor()
            xgb_model.load_model(str(xgb_path))
            xgb_cols = XGB_FEATURES
            if xgb_meta_path.exists():
                meta = json.loads(xgb_meta_path.read_text(encoding="utf-8"))
                xgb_cols = meta.get("X_cols", XGB_FEATURES)
            pred_pr_hist = np.clip(xgb_model.predict(hist_df[xgb_cols]), 0.0, 1.05)
            xgb_hist_mw = pred_pr_hist * budget_hist.values
            b_xgb = cal_factor(hist_df["y_mw"], xgb_hist_mw)
            b_np = (b_xgb + b_exp) / 2.0
            print(
                f"[FORECAST] station {station.pk}: калибровка XGB b_xgb={b_xgb:.3f}, b_np={b_np:.3f}"
            )
        except Exception as e:
            print(f"[FORECAST] station {station.pk}: ошибка калибровки XGB -> {e}")

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
    heur_mw = df_hourly["Expected_MWh_cal"].values

    cap_by_irr_future = budget_future * PR_CEIL_PER_HR

    # === XGB(PR) ===
    xgb_pred_mw = None
    if xgb_path.exists():
        try:
            xgb_model = xgb.XGBRegressor()
            xgb_model.load_model(str(xgb_path))
            xgb_cols = XGB_FEATURES
            if xgb_meta_path.exists():
                meta = json.loads(xgb_meta_path.read_text(encoding="utf-8"))
                xgb_cols = meta.get("X_cols", XGB_FEATURES)
            df_xgb = df_hourly[xgb_cols].copy()
            pred_pr = np.clip(xgb_model.predict(df_xgb), 0.0, 1.05)
            xgb_pred_mw = pred_pr * budget_future.values
            xgb_pred_mw = np.minimum(xgb_pred_mw, np.minimum(cap_by_irr_future.values, np.full_like(cap_by_irr_future.values, cap_mw)))
            print(
                f"[FORECAST] station {station.pk}: XGB(PR) загружен ({xgb_path.name}), строк={len(df_xgb)}"
            )
        except Exception as e:
            print(f"[FORECAST] station {station.pk}: ошибка прогноза XGB -> {e}")

    # === NeuralProphet residual ===
    np_pred_mw = None
    np_path = MODEL_DIR / f"np_model_{station.pk}.np"
    np_meta_path = MODEL_DIR / f"np_model_{station.pk}.meta.json"
    if np_path.exists():
        try:
            np_model = np_load(str(np_path))
            meta_np = {}
            if np_meta_path.exists():
                meta_np = json.loads(np_meta_path.read_text(encoding="utf-8"))
            cap_train_np = float(meta_np.get("cap_mw_train", cap_mw) or cap_mw)
            features_reg = meta_np.get("features_reg", NP_REGRESSORS)

            expected_train_future = (
                cap_train_np * (df_hourly["Irradiation"] / 1000.0) * PR_BASE
            ).clip(0, cap_train_np * 0.95)

            df_np = df_hourly.copy()
            df_np["y_expected_log"] = np.log1p(expected_train_future * 0.95)

            df_np_in = df_np[["ds"] + features_reg].copy()
            df_np_in["y"] = 0.0

            np_forecast = np_model.predict(df_np_in)
            residual_pred = np_forecast["yhat1"].to_numpy()

            cap_ratio_np = cap_mw / cap_train_np if cap_train_np > 0 else 1.0
            np_pred_mw = (expected_future.values + residual_pred * cap_ratio_np).clip(lower=0.0)
            np_pred_mw = np.minimum(np_pred_mw, np.minimum(cap_by_irr_future.values, np.full_like(cap_by_irr_future.values, cap_mw)))
            print(
                f"[FORECAST] station {station.pk}: NP residual загружен ({np_path.name}), строк={len(df_np_in)}"
            )
        except Exception as e:
            print(f"[FORECAST] station {station.pk}: ошибка прогноза NP -> {e}")

    # === Калибровка и клипы ===
    np_pred_mw_cal = None
    xgb_pred_mw_cal = None

    lim_after = np.minimum.reduce(
        [
            cap_by_irr_future.values,
            np.full_like(cap_by_irr_future.values, cap_mw),
            (df_hourly["Expected_MWh_cal"].values * ENSEMBLE_HEADROOM),
        ]
    )

    if np_pred_mw is not None:
        np_pred_mw_cal = np.minimum(np_pred_mw * b_np, lim_after)

    if xgb_pred_mw is not None:
        xgb_pred_mw_cal = np.minimum(xgb_pred_mw * b_xgb, lim_after)

    expected_cal = np.minimum(df_hourly["Expected_MWh_cal"].values, lim_after)

    # === Динамический ансамбль ===
    n = len(df_hourly)
    h = df_hourly["ds"].dt.hour
    irr = df_hourly["Irradiation"].fillna(0)
    low = (irr < 300) | (df_hourly["low_sun_flag"] == 1)

    # базовый фоллбек — откалиброванная эвристика, чтобы не получить UnboundLocal
    ensemble_mw = heur_mw.copy() if heur_mw is not None else np.zeros(n)

    try:
        w_np = np.zeros(n) if np_pred_mw_cal is None else np.where((h.between(6, 9)) & (irr > 60), 0.45, 0.30)
        w_xgb = np.zeros(n) if xgb_pred_mw_cal is None else np.where(h.between(6, 9), 0.35, 0.30)
        w_exp = 1.0 - (w_np + w_xgb)

        bright_midday = (h.between(11, 15)) & (irr > 700)
        w_exp = np.where(bright_midday, np.minimum(w_exp + 0.10, 0.85), w_exp)
        if np_pred_mw_cal is not None:
            w_np = np.where(bright_midday, np.maximum(w_np - 0.05, 0.05), w_np)
        if xgb_pred_mw_cal is not None:
            w_xgb = np.where(bright_midday, np.maximum(w_xgb - 0.05, 0.05), w_xgb)

        w_exp = np.where(low, np.minimum(w_exp + 0.20, 0.90), w_exp)
        if np_pred_mw_cal is not None:
            w_np = np.where(low, np.maximum(w_np - 0.10, 0.05), w_np)
        if xgb_pred_mw_cal is not None:
            w_xgb = np.where(low, np.maximum(w_xgb - 0.10, 0.05), w_xgb)

        wsum = w_np + w_xgb + w_exp
        wsum = np.where(wsum == 0, 1.0, wsum)
        w_np, w_xgb, w_exp = w_np / wsum, w_xgb / wsum, w_exp / wsum

        zeros = np.zeros(n)
        np_safe = np_pred_mw_cal if np_pred_mw_cal is not None else zeros
        xgb_safe = xgb_pred_mw_cal if xgb_pred_mw_cal is not None else zeros

        ensemble_mw_raw = (
            w_np * np_safe +
            w_xgb * xgb_safe +
            w_exp * expected_cal
        )

        ens_limit = np.minimum.reduce(
            [
                cap_by_irr_future.values,
                np.full_like(cap_by_irr_future.values, cap_mw),
                expected_cal * ENSEMBLE_HEADROOM,
            ]
        )
        ensemble_mw = np.minimum(ensemble_mw_raw, ens_limit)
    except Exception as e:
        print(f"[FORECAST] station {station.pk}: fallback на эвристику из-за ошибки ансамбля -> {e}")

    if ensemble_mw is None or len(ensemble_mw) == 0:
        ensemble_mw = heur_mw.copy() if heur_mw is not None else np.zeros(n)

    # Ночной фильтр
    mask_night = (df_hourly["Irradiation"] < 20) | (df_hourly["sun_elev_deg"] < 5)
    heur_mw = np.where(mask_night, 0.0, heur_mw)
    ensemble_mw = np.where(mask_night, 0.0, ensemble_mw)
    if xgb_pred_mw_cal is not None:
        xgb_pred_mw_cal = np.where(mask_night, 0.0, xgb_pred_mw_cal)
    if np_pred_mw_cal is not None:
        np_pred_mw_cal = np.where(mask_night, 0.0, np_pred_mw_cal)

    # === Сохранение в SolarForecast (MW -> кВт) ===
    timestamps = pd.to_datetime(df_hourly["ds"]).tolist()

    with transaction.atomic():
        SolarForecast.objects.filter(
            station=station,
            timestamp__in=timestamps,
        ).delete()

        objs = []
        for idx, (ds, v_exp, v_ens) in enumerate(
            zip(timestamps, heur_mw, ensemble_mw, strict=True),
        ):
            if timezone.is_naive(ds):
                ds = make_aware(ds, timezone.get_default_timezone())

            exp_kw = float(v_exp * 1000.0)
            ens_kw = float(v_ens * 1000.0)
            pred_xgb_kw = (
                float(xgb_pred_mw_cal[idx] * 1000.0) if xgb_pred_mw_cal is not None else 0.0
            )
            pred_np_kw = (
                float(np_pred_mw_cal[idx] * 1000.0) if np_pred_mw_cal is not None else 0.0
            )

            objs.append(
                SolarForecast(
                    station=station,
                    timestamp=ds,
                    pred_np=pred_np_kw,
                    pred_xgb=pred_xgb_kw,
                    pred_heur=exp_kw,     # эвристика (кВт)
                    pred_final=ens_kw,    # итог = динамический ансамбль после клипов
                )
            )

        SolarForecast.objects.bulk_create(objs)

    print(
        f"[FORECAST] station {station.pk}: прогноз записан в SolarForecast, строк: {len(timestamps)}, "
        f"cap_kw={cap_kw:.2f}"
    )
    return len(timestamps)

