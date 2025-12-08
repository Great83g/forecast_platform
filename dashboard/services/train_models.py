# dashboard/services/train_models.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

from django.conf import settings
import numpy as np
import pandas as pd
import xgboost as xgb
from neuralprophet import NeuralProphet, save as np_save

from solar.models import SolarRecord

# Папка с моделями (в backend.settings: MODEL_DIR = BASE_DIR / "models_cache")
MODEL_DIR: Path = Path(settings.MODEL_DIR)

# Базовый PR, как в боевых скриптах
PR_FOR_EXPECTED = 0.90


# =========================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =========================

def _get_station_cap_mw(station, df: pd.DataFrame) -> float:
    """
    Оценка установленной мощности станции в МВт.
    1) station.capacity_kw (если есть),
    2) максимум по истории / 1000,
    3) дефолт 10 МВт.
    """
    cap_kw = getattr(station, "capacity_kw", None)
    if cap_kw:
        try:
            cap_mw = float(cap_kw) / 1000.0
            if cap_mw > 0:
                return cap_mw
        except Exception:
            pass

    max_kw = float(df["Power_KW"].max() or 0.0)
    if max_kw > 0:
        return max_kw / 1000.0

    # запасной дефолт
    return 10.0


def _prepare_history_df(station) -> pd.DataFrame:
    """
    История SolarRecord для станции.
    Выход: ds, Irradiation, Air_Temp, PV_Temp, Power_KW
    (без фич – их добавляем отдельно).
    """
    qs = (
        SolarRecord.objects
        .filter(station=station)
        .order_by("timestamp")
    )
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

    # чистим мусор
    numeric_cols = ["Power_KW", "Irradiation", "Air_Temp", "PV_Temp"]
    before = len(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=numeric_cols)
    removed = before - len(df)
    if removed:
        print(
            f"[TRAIN] station {station.pk}: удалено {removed} строк "
            f"с NaN/inf в {numeric_cols}"
        )

    return df


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
    """
    Фичи 1в1 с твоими скриптами (без cloudcover, её нет в истории):
      hour, month, hour_sin, month_sin,
      is_clear, morning_peak_boost, evening_penalty, overdrive_flag,
      midday_penalty, is_morning_active,
      PV_Temp, y_expected, y_expected_log.
    """
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

    # PV temp proxy
    df["PV_Temp"] = df["Air_Temp"] + np.maximum(df["Irradiation"] - 50, 0) / 1000.0 * 20.0

    # ожидаемая генерация (MW) при PR=0.9
    df["y_expected"] = cap_mw * (df["Irradiation"] / 1000.0) * PR_FOR_EXPECTED
    df["y_expected"] = df["y_expected"].clip(upper=cap_mw * 0.95)
    df["y_expected_log"] = np.log1p(df["y_expected"] * 0.95)

    return df


# =========================
# ОСНОВНАЯ ФУНКЦИЯ ОБУЧЕНИЯ
# =========================

def train_models_for_station(station) -> Tuple[int, Path | None, Path | None]:
    """
    Обучает:
      - XGBoost по мощности на 1 МВт (per-MW)
      - NeuralProphet напрямую по мощности в МВт
    История берётся из SolarRecord.

    Возвращает:
      (n_rows, np_path, xgb_path)
    """
    df = _prepare_history_df(station)
    n_rows = len(df)
    if n_rows == 0:
        print(f"[TRAIN] station {station.pk}: нет данных для обучения")
        return 0, None, None

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # пути для моделей (оставляем старый формат имён)
    np_path = MODEL_DIR / f"np_model_{station.pk}.np"
    np_meta_path = MODEL_DIR / f"np_model_{station.pk}.meta.json"
    xgb_path = MODEL_DIR / f"xgb_model_{station.pk}.json"
    xgb_meta_path = MODEL_DIR / f"xgb_model_{station.pk}.meta.json"

    print(f"[TRAIN] station {station.pk}: MODEL_DIR = {MODEL_DIR}")
    print(f"[TRAIN] station {station.pk}: строк в истории = {n_rows}")

    # единицы: переводим Power_KW -> MW (1 час шаг)
    df["y_mw"] = df["Power_KW"] / 1000.0

    # оценка мощности станции
    cap_mw = _get_station_cap_mw(station, df)
    print(f"[TRAIN] station {station.pk}: cap_mw≈{cap_mw:.3f}")

    # общие фичи
    df = _add_common_features(df, cap_mw=cap_mw)
    df = _add_sun_geometry(df, ds_col="ds", lat_deg=47.86)

    # ==================== 1) XGBoost per-MW ====================
    try:
        X_cols = [
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
        df["y_permw"] = (df["y_mw"] / cap_mw).clip(lower=0)

        df_xgb = df.dropna(subset=X_cols + ["y_permw"]).copy()

        print(
            f"[TRAIN] station {station.pk}: старт обучения XGB(per-MW), "
            f"строк после dropna={len(df_xgb)}"
        )

        xgb_model = xgb.XGBRegressor(
            n_estimators=700,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=42,
            objective="reg:squarederror",
        )
        xgb_model.fit(df_xgb[X_cols], df_xgb["y_permw"])
        xgb_model.save_model(str(xgb_path))
        print(f"[TRAIN] station {station.pk}: XGB(per-MW) сохранён в {xgb_path}")

        xgb_meta = {
            "cap_mw_train": cap_mw,
            "X_cols": X_cols,
            "target": "y_permw",
            "note": "XGB per-MW (predict MW per installed MW)",
        }
        xgb_meta_path.write_text(
            json.dumps(xgb_meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[TRAIN] station {station.pk}: XGB meta сохранён в {xgb_meta_path}")
    except Exception as e:
        import traceback
        print(f"[TRAIN] station {station.pk}: ОШИБКА при обучении XGB(PR) -> {e}")
        traceback.print_exc()
        xgb_path = None
        X_cols = []

    # ==================== 2) NeuralProphet по y_mw ====================
    try:
        df_np = df.copy()
        # целевая: прямая мощность в MW
        df_np["y_target"] = df_np["y_mw"]

        # балансировка утра
        df_np["dup_weight"] = 1
        df_np.loc[df_np["is_morning_active"] == 1, "dup_weight"] = 8

        df_dup = df_np.loc[df_np.index.repeat(df_np["dup_weight"])].copy()
        df_dup["dup_index"] = df_dup.groupby("ds").cumcount()
        df_dup["ds"] = df_dup["ds"] + pd.to_timedelta(df_dup["dup_index"], unit="m")
        df_b = df_dup.drop(columns=["dup_index", "dup_weight"])

        features_reg = [
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

        df_train = df_b[["ds", "y_target"] + features_reg].dropna().copy()
        df_train.rename(columns={"y_target": "y"}, inplace=True)

        print(f"[TRAIN] station {station.pk}: старт обучения NeuralProphet(y), строк={len(df_train)}")

        m = NeuralProphet(
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
        for col in features_reg:
            m.add_future_regressor(col, normalize="minmax")

        m.fit(df_train, freq="H")

        np_save(m, str(np_path))
        print(f"[TRAIN] station {station.pk}: NP(y) сохранён в {np_path}")

        np_meta = {
            "cap_mw_train": cap_mw,
            "pr_for_expected": PR_FOR_EXPECTED,
            "features_reg": features_reg,
            "target": "y (MW) = Power_KW/1000",
            "note": "NP trained on direct y from SolarRecord history",
        }
        np_meta_path.write_text(
            json.dumps(np_meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[TRAIN] station {station.pk}: NP meta сохранён в {np_meta_path}")
    except Exception as e:
        import traceback
        print(f"[TRAIN] station {station.pk}: ОШИБКА при обучении NP(y) -> {e}")
        traceback.print_exc()
        np_path = None

    return n_rows, np_path, xgb_path

