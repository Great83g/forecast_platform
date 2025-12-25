# dashboard/services/forecast_engine.py
from __future__ import annotations

import json
import logging
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
logger = logging.getLogger(__name__)


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


def _describe_np_model(model: object) -> str:
    if model is None:
        return "model=None"
    parts = [
        f"type={type(model)}",
        f"has_predict={hasattr(model, 'predict')}",
        f"has_trainer={getattr(model, 'trainer', None) is not None}",
        f"has_model={getattr(model, 'model', None) is not None}",
        f"has_init_trainer={callable(getattr(model, '_init_trainer', None))}",
    ]
    return ", ".join(parts)


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
    Делает сетку часов на days вперёд (включая завтра), ограничивая "солнечными" часами.
    """
    now = timezone.localtime(timezone.now())
    try:
        h1, h2 = solar_hours
    except Exception:
        logger.warning("[FORECAST] invalid solar_hours=%s, fallback to (5, 20)", solar_hours)
        h1, h2 = 5, 20

    # начинаем с ближайшего следующего дня, чтобы не строить уже прошедшие часы
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

    out["is_clear"] = ((out["Irradiation"] > 200) & (out["Air_Temp"] > 0)).astype(int)

    out["morning_peak_boost"] = ((out["hour"] == 6) & (out["Irradiation"] > 39)).astype(int)
    out["evening_penalty"] = ((out["hour"] == 19) & (out["Irradiation"] > 39)).astype(int)
    out["overdrive_flag"] = ((out["Irradiation"] > 950) & (out["Air_Temp"] > 30)).astype(int)
    out["midday_penalty"] = ((out["hour"].isin([12, 13, 14]))).astype(int)
    out["is_morning_active"] = ((out["hour"] == 6) & (out["Irradiation"] > 49)).astype(int)

    # ожидаемая генерация и лог-таргет (как в обучении)
    expected_mw = (capacity_mw * (out["Irradiation"] / 1000.0) * PR_FOR_EXPECTED).clip(upper=capacity_mw * 0.95)
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


# =========================
# === NeuralProphet FIX ===
# =========================

def _allow_torch_safe_globals_for_np() -> None:
    """
    PyTorch 2.6: safe-unpickle режет pandas/neuralprophet объекты.
    Best-effort allowlist под разные версии pandas.
    """
    try:
        from torch.serialization import add_safe_globals

        from neuralprophet.forecaster import NeuralProphet
        from neuralprophet.configure import Normalization
        from neuralprophet.df_utils import ShiftScale

        from pandas._libs.tslibs import timestamps as _ts
        from pandas._libs.tslibs import timedeltas as _td

        allow = [NeuralProphet, Normalization, ShiftScale]

        # timestamp helper variants
        for name in ("_unpickle_timestamp", "_timestamp_unpickle"):
            fn = getattr(_ts, name, None)
            if fn is not None:
                allow.append(fn)

        # timedelta helper variants (в твоих ошибках часто именно _timedelta_unpickle)
        for name in ("_unpickle_timedelta", "_timedelta_unpickle"):
            fn = getattr(_td, name, None)
            if fn is not None:
                allow.append(fn)

        add_safe_globals(allow)
    except Exception:
        return


def _load_np_model(path: Path):
    """
    Грузим .np через neuralprophet.load() (правильный формат — model.save()).
    Fallback: torch.load(weights_only=False).

    ВАЖНО: если загружается объект NeuralProphet, но obj.model == None,
    то predict() падает 'NoneType has no attribute predict'. Ловим это здесь и даём ясную причину.
    """
    _allow_torch_safe_globals_for_np()
    torch_err: Optional[str] = None
    np_err: Optional[str] = None

    def _extract(m: object) -> Optional[object]:
        if m is None:
            return None
        if hasattr(m, "predict"):
            return m
        if isinstance(m, (tuple, list)):
            for itm in m:
                cand = _extract(itm)
                if cand is not None:
                    return cand
            return None
        if isinstance(m, dict):
            for key in ("model", "forecaster", "np_model", "forecast_model"):
                cand = _extract(m.get(key))
                if cand is not None:
                    return cand
            for v in m.values():
                cand = _extract(v)
                if cand is not None:
                    return cand
            return None
        return None

    def _validate_np(obj: object, src: str) -> object:
        if obj is None or not hasattr(obj, "predict"):
            raise TypeError(f"NP load returned invalid object from {src}: {type(obj)}")

        # КЛЮЧЕВО: внутренний torch-модуль должен быть восстановлен
        inner = getattr(obj, "model", None)
        if inner is None:
            raise TypeError(
                "NeuralProphet loaded but internal `model` is None (weights not restored). "
                "Обычно это значит: файл НЕ настоящий .np от `model.save()`, "
                "или повреждён/не той версии. Пересохрани модель через `model.save('...np')`."
            )
        return obj

    # 1) native NP loader
    try:
        loaded = np_load(str(path))
        model = _extract(loaded)
        if model is not None:
            return _validate_np(model, f"neuralprophet.load({path.name})")
    except Exception as e:
        np_err = str(e)

    # 2) torch fallback
    try:
        import torch
        loaded = torch.load(str(path), map_location="cpu", weights_only=False)
        model = _extract(loaded)
        if model is not None:
            return _validate_np(model, f"torch.load({path.name})")
    except Exception as e:
        torch_err = str(e)

    raise TypeError(f"NP load failed: np_err={np_err}, torch_err={torch_err}")


def _predict_np(
    model,
    df_feat: pd.DataFrame,
    reg_features: Optional[List[str]] = None,
    cap_for_expected: Optional[float] = None,
) -> np.ndarray:
    """
    Предикт NeuralProphet:
    - model.predict ожидает df с 'ds' и будущими регрессорами, если они были при обучении.
    Тут мы подаём минимум: ds + регрессоры Irradiation/Air_Temp/PV_Temp и т.п.
    """
    if model is None or not hasattr(model, "predict"):
        raise TypeError("NP model is not loaded or has no predict() method")

    if getattr(model, "trainer", None) is None:
        init_trainer = getattr(model, "_init_trainer", None)
        restore_trainer = getattr(model, "restore_trainer", None)
        errors: List[str] = []
        if callable(restore_trainer):
            try:
                trainer_obj = restore_trainer()
                if trainer_obj is not None and getattr(model, "trainer", None) is None:
                    model.trainer = trainer_obj
            except Exception as exc:
                errors.append(f"restore_trainer: {exc}")
        if getattr(model, "trainer", None) is None and callable(init_trainer):
            try:
                trainer_obj = init_trainer()
                if trainer_obj is not None and getattr(model, "trainer", None) is None:
                    model.trainer = trainer_obj
            except TypeError as exc:
                errors.append(f"default: {exc}")
                try:
                    trainer_obj = init_trainer(max_epochs=1)
                    if trainer_obj is not None and getattr(model, "trainer", None) is None:
                        model.trainer = trainer_obj
                except Exception as exc2:
                    errors.append(f"max_epochs=1: {exc2}")
            except Exception as exc:
                errors.append(f"default: {exc}")
        if getattr(model, "trainer", None) is None:
            details = f" Ошибка инициализации: {', '.join(errors)}" if errors else ""
            logger.warning(
                "[NP] NeuralProphet loaded without trainer (predict cannot run). "
                "Пересохрани модель через `model.save('...np')` или переобучи.%s | %s",
                details,
                _describe_np_model(model),
            )
            return np.full(len(df_feat), np.nan)

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

    allowed_regressors = None
    if hasattr(model, "config_regressors"):
        config_regs = getattr(model, "config_regressors", None)
        regs = getattr(config_regs, "regressors", None) if config_regs is not None else None
        if isinstance(regs, dict):
            allowed_regressors = list(regs.keys())

    if allowed_regressors is not None:
        unexpected = [c for c in reg_list if c not in allowed_regressors]
        if unexpected:
            logger.warning("[NP] regressors not in model config -> dropped: %s", unexpected)
        reg_list = allowed_regressors

    # если нет y_expected_log, посчитаем на основе irradiation и мощности
    if "y_expected_log" not in df_feat.columns and "Irradiation" in df_feat.columns:
        cap_use = float(cap_for_expected) if cap_for_expected is not None else 1.0
        expected_mw = (cap_use * (df_feat["Irradiation"] / 1000.0) * PR_FOR_EXPECTED).clip(0, cap_use * 0.95)
        df_feat["y_expected_log"] = np.log1p(expected_mw)

    dfp = pd.DataFrame({"ds": pd.to_datetime(df_feat["ds"])})
    # y нужен для некоторых версий NP даже в будущем — кладём NaN
    dfp["y"] = np.nan

    missing = []
    for col in reg_list:
        if col in df_feat.columns:
            dfp[col] = df_feat[col].values
        else:
            missing.append(col)
            dfp[col] = 0.0

    if missing:
        logger.warning("[NP] missing regressors -> filled with 0.0: %s", missing)

    fcst = model.predict(dfp)

    # NeuralProphet обычно возвращает yhat1
    yhat_col = "yhat1" if "yhat1" in fcst.columns else None
    if not yhat_col:
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
    p = (irr / 1000.0) * capacity_mw
    p = np.clip(p, 0, capacity_mw)
    return p


@transaction.atomic
def run_forecast_for_station(station_id: int, days: int = 1) -> Dict:
    st = Station.objects.get(pk=station_id)
    capacity_mw = _station_capacity_mw(st)
    solar_hours = _solar_hours_from_history(st)

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

    if not np_path.exists() or not xgb_path.exists():
        try:
            from .train_models import train_models_for_station

            logger.info(
                "[MODEL] missing model files (np=%s, xgb=%s). Attempting auto-train.",
                np_path.exists(),
                xgb_path.exists(),
            )
            _, np_path_new, xgb_path_new = train_models_for_station(st)
            if np_path_new is not None:
                np_path = np_path_new
            if xgb_path_new is not None:
                xgb_path = xgb_path_new
        except Exception as exc:
            logger.exception("[MODEL] auto-train failed: %s", exc)
        else:
            if np_meta_path.exists():
                try:
                    np_meta = json.loads(np_meta_path.read_text(encoding="utf-8"))
                except Exception:
                    np_meta = {}
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
    booster = None
    if xgb_path.exists():
        booster = _load_xgb_model(xgb_path)
        if booster is None:
            xgb_error = f"XGB load failed: {xgb_path}"
            logger.warning("[XGB] load failed from %s", xgb_path)
    elif abs(capacity_mw - 8.8) < 0.05 and fallback_xgb_path.exists():
        booster = _load_xgb_model(fallback_xgb_path)
        if booster is None:
            xgb_error = f"XGB load failed: {fallback_xgb_path}"
            logger.warning("[XGB] load failed from %s", fallback_xgb_path)
        if fallback_xgb_meta_path.exists():
            try:
                xgb_meta = json.loads(fallback_xgb_meta_path.read_text(encoding="utf-8"))
            except Exception:
                xgb_meta = xgb_meta
    else:
        xgb_error = f"XGB model not found: {xgb_path}"
        logger.warning("[XGB] model not found: %s", xgb_path)

    if booster is not None:
        try:
            feature_names = xgb_meta.get("X_cols") or XGB_EXPECTED_FEATURES
            y_xgb = _predict_xgb(booster, feat, feature_names)
            xgb_ok = True
        except Exception as e:
            xgb_error = str(e)
            xgb_ok = False
            booster = None

    # NP (FIXED)
    if np_path.exists():
        try:
            model = _load_np_model(np_path)
            logger.info("[NP] loaded from %s %s", np_path, _describe_np_model(model))
            y_np = _predict_np(
                model,
                feat,
                reg_features=np_meta.get("features_reg"),
                cap_for_expected=np_meta.get("cap_mw_used"),
            )
            np_ok = True
        except Exception as e:
            logger.exception("[NP] ERROR: %s", e)
            np_error = str(e)
            np_ok = False
    elif abs(capacity_mw - 8.8) < 0.05 and fallback_np_path.exists():
        try:
            model = _load_np_model(fallback_np_path)
            if fallback_np_meta_path.exists():
                try:
                    np_meta = json.loads(fallback_np_meta_path.read_text(encoding="utf-8"))
                except Exception:
                    np_meta = np_meta
            logger.info("[NP] loaded from %s %s", fallback_np_path, _describe_np_model(model))
            y_np = _predict_np(
                model,
                feat,
                reg_features=np_meta.get("features_reg"),
                cap_for_expected=np_meta.get("cap_mw") or np_meta.get("cap_mw_used"),
            )
            np_ok = True
        except Exception as e:
            logger.exception("[NP] ERROR: %s", e)
            np_error = str(e)
            np_ok = False
    else:
        np_error = f"NP model not found: {np_path}"
        logger.warning("[NP] model not found: %s", np_path)

    # эвристика (MW)
    y_heur = _heuristic_mw(feat, capacity_mw=capacity_mw)

    # ансамбль:
    y_final = y_heur.copy()
    if xgb_ok:
        y_final = 0.6 * y_heur + 0.4 * y_xgb
    if np_ok and xgb_ok:
        y_final = 0.2 * y_heur + 0.4 * y_xgb + 0.4 * y_np
    elif np_ok and not xgb_ok:
        y_final = 0.6 * y_heur + 0.4 * y_np

    # клип по мощности станции (MW) и перевод в кВт для сохранения
    y_np = np.clip(np.nan_to_num(y_np, nan=0.0), 0, capacity_mw)
    y_xgb = np.clip(np.nan_to_num(y_xgb, nan=0.0), 0, capacity_mw)
    y_heur = np.clip(np.nan_to_num(y_heur, nan=0.0), 0, capacity_mw)
    y_final = np.clip(np.nan_to_num(y_final, nan=0.0), 0, capacity_mw)

    y_np_kw = y_np * 1000.0
    y_xgb_kw = y_xgb * 1000.0
    y_heur_kw = y_heur * 1000.0
    y_final_kw = y_final * 1000.0

    # ---- save ----
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
