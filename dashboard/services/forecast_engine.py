# dashboard/services/forecast_engine.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from django.conf import settings
from django.db import transaction
from django.utils import timezone

from neuralprophet import load as np_load

from solar.models import SolarForecast, SolarRecord
from stations.models import Station
from .vc_weather import fetch_visual_crossing_hourly


MODEL_DIR: Path = Path(getattr(settings, "MODEL_DIR", Path(settings.BASE_DIR) / "models_cache"))


XGB_EXPECTED_FEATURES = [
    "Irradiation",
    "Air_Temp",
    "PV_Temp",
    "hour",
    "month",
    "hour_sin",
    "month_sin",
    "sun_elev_deg",
    "low_sun_flag",
    "is_daylight",
    "is_clear",
    "morning_peak_boost",
    "evening_penalty",
    "is_morning_active",
    "overdrive_flag",
    "midday_penalty",
]

PR_FOR_EXPECTED = 0.90


def _station_capacity_mw(st: Station) -> float:
    """
    Пытаемся достать мощность станции.
    Поддерживаем разные поля (потому что у тебя модели/миграции менялись).
    """
    for name in ["capacity_mw", "capacity_ac_mw"]:
        if hasattr(st, name) and getattr(st, name):
            return float(getattr(st, name))

    for name in ["capacity_ac_kw", "capacity_kw", "capacity_dc_kw"]:
        if hasattr(st, name) and getattr(st, name):
            return float(getattr(st, name)) / 1000.0

    # fallback: если нет поля — пусть будет 10MW, чтобы не было микроскопии
    return 10.0


def _solar_hours_from_history(st: Station) -> Tuple[int, int]:
    """
    Берём “солнечные часы” из истории:
    - ищем часы, где irradiation>50 или power_kw>0
    - берём min/max hour
    Всегда гарантируем широкий диапазон 5-20.
    """
    qs = SolarRecord.objects.filter(station=st).order_by("-timestamp")[:14 * 24]
    if not qs.exists():
        return (5, 20)

    df = pd.DataFrame.from_records(qs.values("timestamp", "irradiation", "power_kw"))
    if df.empty:
        return (5, 20)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    mask = (df["irradiation"].fillna(0) > 50) | (df["power_kw"].fillna(0) > 0)
    if mask.sum() < 5:
        return (5, 20)

    hmin = int(df.loc[mask, "hour"].min())
    hmax = int(df.loc[mask, "hour"].max())
    # немного расширим; если окно узкое — берём фиксированный день 5-20
    h1 = max(5, hmin - 1)
    h2 = min(20, hmax + 1)
    if (h2 - h1) < 12:
        return (5, 20)
    return (h1, h2)


def _make_base_grid(days: int, solar_hours: Tuple[int, int]) -> pd.DataFrame:
    """
    Делает сетку часов на days вперёд (включая сегодня/завтра, но только солнечные часы).
    """
    now = timezone.localtime(timezone.now())
    h1, h2 = solar_hours

    # начинаем с ближайшего следующего солнечного дня, чтобы не строить уже прошедшие часы
    start_date = (now + pd.Timedelta(days=1)).date()
    start = (
        timezone.datetime.combine(start_date, timezone.datetime.min.time())
        .replace(hour=h1, tzinfo=now.tzinfo, minute=0, second=0, microsecond=0)
    )
    end = start + pd.Timedelta(days=days)

    all_hours = pd.date_range(start=start, end=end, freq="h", inclusive="left")
    df = pd.DataFrame({"ds": all_hours})

    df = df[(df["ds"].dt.hour >= h1) & (df["ds"].dt.hour <= h2)].copy()
    df["ds"] = df["ds"].dt.floor("h")
    return df.reset_index(drop=True)


def _merge_weather(base: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    w = weather.copy()
    w["ds"] = pd.to_datetime(w["ds"]).dt.floor("h")
    base["ds"] = pd.to_datetime(base["ds"]).dt.floor("h")
    out = base.merge(w, on="ds", how="left")
    return out


def _add_sun_geometry(df: pd.DataFrame, lat_deg: float) -> pd.DataFrame:
    lat = np.deg2rad(lat_deg)
    doy = df["ds"].dt.dayofyear
    hour = df["ds"].dt.hour
    hour_angle = np.deg2rad((hour - 12) * 15)
    decl = np.deg2rad(23.44) * np.sin(2 * np.pi * (284 + doy) / 365)
    sin_elev = (
        np.sin(lat) * np.sin(decl)
        + np.cos(lat) * np.cos(decl) * np.cos(hour_angle)
    )
    df["sun_elev_deg"] = np.rad2deg(np.arcsin(np.clip(sin_elev, -1, 1)))
    df["low_sun_flag"] = (df["sun_elev_deg"] < 15).astype(int)
    return df


def _compute_features(df: pd.DataFrame, capacity_mw: float, lat_deg: float) -> pd.DataFrame:
    """
    Генерим фичи под XGB ожидаемый набор.
    """
    out = df.copy()

    # нормальные имена для XGB
    out["Irradiation"] = pd.to_numeric(out.get("irradiation"), errors="coerce").fillna(0.0)
    out["Air_Temp"] = pd.to_numeric(out.get("air_temp"), errors="coerce").fillna(0.0)

    # PV_Temp — если нет в погоде, аппроксимируем как в локальном скрипте
    out["PV_Temp"] = out["Air_Temp"] + np.maximum(out["Irradiation"] - 50, 0) / 1000 * 20

    out["hour"] = pd.to_datetime(out["ds"]).dt.hour.astype(int)
    out["month"] = pd.to_datetime(out["ds"]).dt.month.astype(int)

    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12.0)

    # простые флаги
    out["is_daylight"] = (out["Irradiation"] > 20).astype(int)

    cloud = pd.to_numeric(out.get("cloudcover"), errors="coerce").fillna(100.0)
    out["is_clear"] = ((cloud < 50) & (out["Irradiation"] > 80)).astype(int)

    out["morning_peak_boost"] = ((out["hour"] == 6) & (out["Irradiation"] > 39)).astype(int)
    out["evening_penalty"] = ((out["hour"] == 19) & (out["Irradiation"] > 39)).astype(int)
    out["overdrive_flag"] = ((out["Irradiation"] > 700) & (out["Air_Temp"] > 25)).astype(int)
    out["midday_penalty"] = ((out["hour"].isin([12, 13, 14])) & (out["Irradiation"] > 600)).astype(int)
    out["is_morning_active"] = ((out["hour"] == 6) & (out["Irradiation"] > 49)).astype(int)

    # ожидаемая генерация и лог-таргет (как в обучении)
    expected_mw = (capacity_mw * (out["Irradiation"] / 1000.0) * PR_FOR_EXPECTED).clip(0, capacity_mw * 0.95)
    out["y_expected_log"] = np.log1p(expected_mw * 0.95)

    out = _add_sun_geometry(out, lat_deg)

    # гарантируем порядок и наличие
    for c in XGB_EXPECTED_FEATURES:
        if c not in out.columns:
            out[c] = 0.0

    return out


def _load_xgb_model(path: Path) -> Optional[xgb.Booster]:
    try:
        booster = xgb.Booster()
        booster.load_model(str(path))
        return booster
    except Exception:
        return None


def _allow_torch_safe_globals_for_np() -> None:
    """
    PyTorch 2.6 включил weights_only=True по умолчанию и начал блокировать классы.
    Мы разрешаем те классы NeuralProphet, которые вылезают в твоих ошибках.
    """
    try:
        import torch
        from torch.serialization import add_safe_globals

        # импорт learnable классов
        import neuralprophet
        from neuralprophet.forecaster import NeuralProphet
        from neuralprophet.configure import Normalization
        from neuralprophet.df_utils import ShiftScale
        from pandas._libs.tslibs import timestamps as _ts
        from pandas._libs.tslibs import timedeltas as _td

        allow = [
            NeuralProphet,
            Normalization,
            ShiftScale,
            _ts._unpickle_timestamp,
            _td._unpickle_timedelta,
        ]

        # иногда вылезает через модульные пути
        add_safe_globals(allow)

    except Exception:
        # если нет torch / другое окружение — молча
        return


def _load_np_model(path: Path):
    """
    Грузим .np через neuralprophet.load().
    Для PyTorch 2.6 делаем allowlist.
    """
    _allow_torch_safe_globals_for_np()
    torch_err = None
    np_err = None
    model = None

    def _extract(m: object) -> object:
        if isinstance(m, (tuple, list)) and m:
            for itm in m:
                candidate = _extract(itm)
                if candidate is not None and hasattr(candidate, "predict"):
                    return candidate
            for itm in m:
                if hasattr(itm, "predict"):
                    return itm
            return m[0]

        if isinstance(m, dict):
            for key in ("model", "forecaster", "np_model", "forecast_model"):
                candidate = m.get(key)
                if candidate is not None and hasattr(candidate, "predict"):
                    return candidate
            for candidate in m.values():
                if candidate is not None and hasattr(candidate, "predict"):
                    return candidate
            return m.get("model") or m.get("forecaster")
        return m

    # Сначала пробуем native loader NeuralProphet
    try:
        model = _extract(np_load(str(path)))
        if model is not None and hasattr(model, "predict"):
            return model
    except Exception as e:
        np_err = str(e)
        model = None

    # Fallback: torch.load с weights_only=False (для старых .np)
    try:
        import torch

        model = _extract(torch.load(str(path), map_location="cpu", weights_only=False))
        if model is not None and hasattr(model, "predict"):
            return model
    except Exception as e:
        torch_err = str(e)
        model = None

    if model is None:
        raise TypeError(f"NP load failed: np_err={np_err}, torch_err={torch_err}")
    raise TypeError(f"Loaded NP object has no predict(): type={type(model)} np_err={np_err} torch_err={torch_err}")

    if model is None:
        raise TypeError(f"NP load failed: np_err={np_err}, torch_err={torch_err}")
    raise TypeError(f"Loaded NP object has no predict(): type={type(model)} np_err={np_err} torch_err={torch_err}")

def _predict_np(model, df_feat: pd.DataFrame, reg_features: Optional[List[str]] = None, cap_for_expected: Optional[float] = None) -> np.ndarray:
    """
    Предикт NeuralProphet:
    - model.predict ожидает df с 'ds' и будущими регрессорами, если они были при обучении.
    Тут мы подаём минимум: ds + регрессоры Irradiation/Air_Temp/PV_Temp и т.п.
    Если модель обучалась на другом наборе — она сама скажет ошибку.
    """
    if model is None or not hasattr(model, "predict"):
        raise TypeError("NP model is not loaded or has no predict() method")
    df_feat = df_feat.copy()

    reg_list = reg_features or [
        "Irradiation",
        "Air_Temp",
        "PV_Temp",
        "hour_sin",
        "month_sin",
        "is_daylight",
        "is_clear",
        "morning_peak_boost",
        "overdrive_flag",
        "midday_penalty",
        "y_expected_log",
    ]

    # если нет y_expected_log, посчитаем на основе irradiation и мощности
    if "y_expected_log" not in df_feat.columns and "Irradiation" in df_feat.columns:
        cap_use = cap_for_expected if cap_for_expected is not None else 1.0
        expected_mw = (cap_use * (df_feat["Irradiation"] / 1000.0) * PR_FOR_EXPECTED).clip(0, cap_use * 0.95)
        df_feat["y_expected_log"] = np.log1p(expected_mw)

    dfp = pd.DataFrame({"ds": pd.to_datetime(df_feat["ds"])})
    # y нужен для некоторых версий NP даже в будущем — кладём NaN
    dfp["y"] = np.nan
    # пробуем подложить все регрессоры из meta/по умолчанию
    for col in reg_list:
        if col in df_feat.columns:
            dfp[col] = df_feat[col].values
        else:
            dfp[col] = 0.0

    fcst = model.predict(dfp)
    # NeuralProphet обычно возвращает yhat1
    yhat_col = "yhat1" if "yhat1" in fcst.columns else None
    if not yhat_col:
        # fallback: первый yhat
        yhat_cols = [c for c in fcst.columns if c.startswith("yhat")]
        yhat_col = yhat_cols[0] if yhat_cols else None

    if not yhat_col:
        return np.full(len(dfp), np.nan)

    return pd.to_numeric(fcst[yhat_col], errors="coerce").to_numpy()


def _predict_xgb(booster: xgb.Booster, df_feat: pd.DataFrame, feature_names: List[str]) -> np.ndarray:
    X = df_feat[feature_names].astype(float)
    dmat = xgb.DMatrix(X, feature_names=feature_names)
    pred = booster.predict(dmat)
    return pred


def _heuristic_mw(df_feat: pd.DataFrame, capacity_mw: float) -> np.ndarray:
    """
    Простая эвристика: мощность ~ irradiation/1000 * capacity * k
    k подбираем грубо, чтобы не было микроскопии.
    """
    irr = df_feat["Irradiation"].astype(float).to_numpy()
    # нормализация irradiation: 0..1000 W/m2
    p = (irr / 1000.0) * capacity_mw
    # лёгкая “кривая” — чтобы утро/вечер не были нулём
    p = np.clip(p, 0, capacity_mw)
    return p


@transaction.atomic
def run_forecast_for_station(station_id: int, days: int = 1) -> Dict:
    st = Station.objects.get(pk=station_id)
    capacity_mw = _station_capacity_mw(st)
    # форсируем широкий диапазон солнечных часов, чтобы покрыть весь день
    solar_hours = (5, 20)

    base = _make_base_grid(days=days, solar_hours=solar_hours)

    # ---- погода ----
    weather_source = "fallback_zero"
    weather_df = pd.DataFrame(columns=["ds", "irradiation", "air_temp", "wind_speed", "cloudcover", "humidity", "precip"])

    lat = getattr(st, "lat", None) or getattr(st, "latitude", None)
    lon = getattr(st, "lon", None) or getattr(st, "longitude", None)

    if lat is not None and lon is not None:
        wres = fetch_visual_crossing_hourly(float(lat), float(lon), days=days)
        if wres.ok and not wres.df.empty:
            weather_source = wres.source
            weather_df = wres.df.copy()

    merged = _merge_weather(base, weather_df)
    lat_deg = float(lat) if lat is not None else 47.86
    feat = _compute_features(merged, capacity_mw, lat_deg)

    # ---- load models ----
    np_path = MODEL_DIR / f"np_model_{station_id}.np"
    xgb_path = MODEL_DIR / f"xgb_model_{station_id}.json"
    np_meta_path = MODEL_DIR / f"np_model_{station_id}.meta.json"
    xgb_meta_path = MODEL_DIR / f"xgb_model_{station_id}.meta.json"
    np_meta: Dict = {}
    if np_meta_path.exists():
        try:
            np_meta = json.loads(np_meta_path.read_text(encoding="utf-8"))
        except Exception:
            np_meta = {}
    xgb_meta: Dict = {}
    if xgb_meta_path.exists():
        try:
            xgb_meta = json.loads(xgb_meta_path.read_text(encoding="utf-8"))
        except Exception:
            xgb_meta = {}

    fallback_np_path = MODEL_DIR / "np_model_1.np"
    fallback_np_meta_path = MODEL_DIR / "np_model_1.meta.json"
    fallback_xgb_path = MODEL_DIR / "xgb_model_1.json"
    fallback_xgb_meta_path = MODEL_DIR / "xgb_model_1.meta.json"
    np_ok = False
    xgb_ok = False
    np_error = None
    xgb_error = None

    y_np = np.full(len(feat), np.nan)
    y_xgb = np.full(len(feat), np.nan)

    # XGB
    xgb_candidates: List[Tuple[Path, Path]] = [(xgb_path, xgb_meta_path)]
    if abs(capacity_mw - 8.8) < 0.05:
        xgb_candidates.append((fallback_xgb_path, fallback_xgb_meta_path))

    for model_path, meta_path in xgb_candidates:
        if not model_path.exists():
            continue
        booster = _load_xgb_model(model_path)
        if booster is None:
            continue
        if meta_path.exists():
            try:
                xgb_meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                xgb_meta = xgb_meta
        try:
            feature_names = xgb_meta.get("X_cols") or XGB_EXPECTED_FEATURES
            y_xgb = _predict_xgb(booster, feat, feature_names)
            xgb_ok = True
            break
        except Exception as e:
            xgb_error = str(e)
            xgb_ok = False
            booster = None

    if not xgb_ok and xgb_error is None:
        xgb_error = f"XGB model not found: {xgb_path}"

    # NP
    np_candidates: List[Tuple[Path, Path]] = [(np_path, np_meta_path)]
    if abs(capacity_mw - 8.8) < 0.05:
        np_candidates.append((fallback_np_path, fallback_np_meta_path))

    for model_path, meta_path in np_candidates:
        if not model_path.exists():
            continue
        try:
            model = _load_np_model(model_path)
            print(f"[NP] loaded={type(model)} has_predict={hasattr(model, 'predict')}")
            if model is None or not hasattr(model, "predict"):
                raise TypeError("NP model is not loaded or has no predict() method")
            if meta_path.exists():
                try:
                    np_meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception:
                    np_meta = np_meta
            y_np = _predict_np(
                model,
                feat,
                reg_features=np_meta.get("features_reg"),
                cap_for_expected=np_meta.get("cap_mw") or np_meta.get("cap_mw_used"),
            )
            np_ok = True
            break
        except Exception as e:
            print(f"[NP] ERROR: {e}")
            np_error = str(e)
            np_ok = False

    if not np_ok and np_error is None:
        np_error = f"NP model not found: {np_path}"

    # эвристика (MW)
    y_heur = _heuristic_mw(feat, capacity_mw=capacity_mw)

    if xgb_ok:
        cap_scale = xgb_meta.get("cap_mw_used") or xgb_meta.get("cap_mw") or capacity_mw
        y_xgb = y_xgb * float(cap_scale)

    y_np = np.clip(np.nan_to_num(y_np, nan=0.0), 0, capacity_mw)
    y_xgb = np.clip(np.nan_to_num(y_xgb, nan=0.0), 0, capacity_mw)
    y_heur = np.clip(np.nan_to_num(y_heur, nan=0.0), 0, capacity_mw)

    # ансамбль:
    # - если NP есть → 0.4 NP + 0.4 XGB + 0.2 эвристика
    # - если NP нет → XGB если есть, иначе эвристика
    y_final = y_heur.copy()
    if xgb_ok:
        y_final = 0.6 * y_heur + 0.4 * y_xgb
    if np_ok and xgb_ok:
        y_final = 0.2 * y_heur + 0.4 * y_xgb + 0.4 * y_np
    elif np_ok and not xgb_ok:
        y_final = 0.6 * y_heur + 0.4 * y_np

    # клип по мощности станции (MW) и перевод в кВт для сохранения
    y_final = np.clip(np.nan_to_num(y_final, nan=0.0), 0, capacity_mw)

    y_np_kw = y_np * 1000.0
    y_xgb_kw = y_xgb * 1000.0
    y_heur_kw = y_heur * 1000.0
    y_final_kw = y_final * 1000.0

    # ---- save ----
    # чистим прогноз на этот диапазон (солнечные часы текущих days)
    ts_min = feat["ds"].min()
    ts_max = feat["ds"].max()
    SolarForecast.objects.filter(station=st, timestamp__gte=ts_min, timestamp__lte=ts_max).delete()

    objs: List[SolarForecast] = []
    for i, row in feat.iterrows():
        objs.append(
            SolarForecast(
                station=st,
                timestamp=pd.to_datetime(row["ds"]).to_pydatetime(),
                # Сохраняем в кВт (модель работает в MW, перевели выше)
                pred_np=float(y_np_kw[i]),
                pred_xgb=float(y_xgb_kw[i]),
                pred_heur=float(y_heur_kw[i]),
                pred_final=float(y_final_kw[i]),
                irradiation_fc=float(row.get("irradiation") or 0.0) if not pd.isna(row.get("irradiation")) else None,
                air_temp_fc=float(row.get("air_temp") or 0.0) if not pd.isna(row.get("air_temp")) else None,
                wind_speed_fc=float(row.get("wind_speed") or 0.0) if not pd.isna(row.get("wind_speed")) else None,
                cloudcover_fc=float(row.get("cloudcover") or 0.0) if not pd.isna(row.get("cloudcover")) else None,
                humidity_fc=float(row.get("humidity") or 0.0) if not pd.isna(row.get("humidity")) else None,
                precip_fc=float(row.get("precip") or 0.0) if not pd.isna(row.get("precip")) else None,
            )
        )

    SolarForecast.objects.bulk_create(objs, batch_size=500)

    return {
        "ok": True,
        "count": len(objs),
        "days": days,
        "solar_hours": list(solar_hours),
        "weather_source": weather_source,
        "np_ok": np_ok,
        "xgb_ok": xgb_ok,
        "np_error": np_error,
        "xgb_error": xgb_error,
    }
