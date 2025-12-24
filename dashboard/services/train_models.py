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

# Параметры expected/PR
PR_FOR_EXPECTED = 0.90


def get_history_dataframe(station) -> pd.DataFrame:
    """
    История по станции из SolarRecord.

    Возвращает DataFrame с колонками:
      ds, Power_KW, Irradiation, Air_Temp, PV_Temp
    """
    qs = (
        SolarRecord.objects.filter(station=station)
        .order_by("timestamp")
        .values("timestamp", "power_kw", "irradiation", "air_temp", "pv_temp")
    )
    df = pd.DataFrame.from_records(qs)
    if df.empty:
        print(f"[TRAIN] station {station.pk}: история пуста")
        return df

    df.rename(
        columns={
            "timestamp": "ds",
            "power_kw": "Power_KW",
            "irradiation": "Irradiation",
            "air_temp": "Air_Temp",
            "pv_temp": "PV_Temp",
        },
        inplace=True,
    )
    df["ds"] = pd.to_datetime(df["ds"])

    numeric_cols = ["Power_KW", "Irradiation", "Air_Temp"]
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


def compute_cap_mw(df: pd.DataFrame) -> float:
    """
    Грубая оценка установленной мощности станции (MW) по истории.
    """
    if df.empty:
        return 1.0
    max_kw = float(df["Power_KW"].max())
    # минимальный кап 0.5 МВт
    return max(0.5, max_kw / 1000.0)


def station_capacity_mw(station, df: pd.DataFrame) -> float:
    for name in ["capacity_mw", "capacity_ac_mw"]:
        if hasattr(station, name) and getattr(station, name):
            return float(getattr(station, name))

    for name in ["capacity_ac_kw", "capacity_kw", "capacity_dc_kw"]:
        if hasattr(station, name) and getattr(station, name):
            return float(getattr(station, name)) / 1000.0

    return compute_cap_mw(df)


def add_sun_geometry(df: pd.DataFrame, ds_col: str = "ds", lat_deg: float = 47.86) -> pd.DataFrame:
    lat = np.deg2rad(lat_deg)
    doy = df[ds_col].dt.dayofyear
    hour = df[ds_col].dt.hour
    hour_angle = np.deg2rad((hour - 12) * 15)
    decl = np.deg2rad(23.44) * np.sin(2 * np.pi * (284 + doy) / 365)
    sin_elev = (
        np.sin(lat) * np.sin(decl)
        + np.cos(lat) * np.cos(decl) * np.cos(hour_angle)
    )
    df["sun_elev_deg"] = np.rad2deg(np.arcsin(np.clip(sin_elev, -1, 1)))
    df["low_sun_flag"] = (df["sun_elev_deg"] < 15).astype(int)
    return df


def add_common_features(df: pd.DataFrame, cap_mw: float, ds_col: str = "ds") -> pd.DataFrame:
    """
    Добавляем все фичи, которые нужны и для XGB, и для NeuralProphet.

    На выходе есть:
      hour, month, hour_sin, month_sin,
      is_clear, morning_peak_boost, evening_penalty,
      overdrive_flag, midday_penalty, is_morning_active,
      y_expected_log
    """
    df = df.copy()
    df["ds"] = pd.to_datetime(df["ds"])

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

    df["PV_Temp"] = df["Air_Temp"] + np.maximum(df["Irradiation"] - 50, 0) / 1000 * 20

    df["y_expected"] = cap_mw * (df["Irradiation"] / 1000) * PR_FOR_EXPECTED
    df["y_expected"] = df["y_expected"].clip(upper=cap_mw * 0.95)
    df["y_expected_log"] = np.log1p(df["y_expected"] * 0.95)

    return df


def train_models_for_station(station) -> Tuple[int, Path | None, Path | None]:
    """
    Обучаем XGB(per-MW) и NeuralProphet(y) для одной станции.

    Сохраняем:
      - XGB:  xgb_model_{pk}.json + .meta.json
      - NP:   np_model_{pk}.np   + .meta.json

    Возвращаем: (кол-во строк истории, путь к NP, путь к XGB)
    """
    if not hasattr(station, "pk"):
        from stations.models import Station

        station = Station.objects.get(pk=station)

    df = get_history_dataframe(station)
    if df.empty:
        return 0, None, None

    cap_mw = station_capacity_mw(station, df)
    print(f"[TRAIN] station {station.pk}: оценка cap_mw={cap_mw:.3f}")

    lat_deg = getattr(station, "lat", None) or getattr(station, "latitude", None) or 47.86

    df["y"] = (df["Power_KW"] / 1000.0).clip(lower=0)
    df = df[df["y"] >= 0].copy()

    df = add_common_features(df, cap_mw, "ds")
    df = add_sun_geometry(df, "ds", float(lat_deg))
    n_rows = len(df)

    # =========================================================
    # 1) Обучение XGB per-MW
    # =========================================================
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

    df["y_permw"] = (df["y"] / cap_mw).clip(lower=0)
    df_xgb = df.dropna(subset=X_cols + ["y_permw"]).copy()

    xgb_path: Path | None = None
    if len(df_xgb) == 0:
        print(f"[TRAIN] station {station.pk}: нет строк для XGB после фильтрации")
    else:
        try:
            model_xgb = xgb.XGBRegressor(
                n_estimators=500,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
            )
            model_xgb.fit(df_xgb[X_cols], df_xgb["y_permw"])

            xgb_path = MODEL_DIR / f"xgb_model_{station.pk}.json"
            model_xgb.save_model(str(xgb_path))

            meta_xgb = {
                "station_id": station.pk,
                "X_cols": X_cols,
                "cap_mw_used": cap_mw,
                "target": "y_per_MW = y / cap_mw",
            }
            meta_path = MODEL_DIR / f"xgb_model_{station.pk}.meta.json"
            meta_path.write_text(
                json.dumps(meta_xgb, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(
                f"[TRAIN] station {station.pk}: XGB(per-MW) обучен, "
                f"сохранён в {xgb_path}"
            )
        except Exception as e:
            import traceback

            print(f"[TRAIN] station {station.pk}: ОШИБКА при обучении XGB -> {e}")
            traceback.print_exc()
            xgb_path = None

    # =========================================================
    # 2) Обучение NeuralProphet на y_mw
    # =========================================================
    np_path: Path | None = None
    try:
        df_np = df.copy()
        df_np["dup_weight"] = 1
        df_np.loc[df_np["is_morning_active"] == 1, "dup_weight"] = 8
        df_dup = df_np.loc[df_np.index.repeat(df_np["dup_weight"])].copy()
        df_dup["dup_index"] = df_dup.groupby("ds").cumcount()
        df_dup["ds"] = df_dup["ds"] + pd.to_timedelta(df_dup["dup_index"], unit="m")
        df_b = df_dup.drop(columns=["dup_index", "dup_weight"])

        print(
            f"[TRAIN] station {station.pk}: старт обучения NP(y), строк={len(df_b)}"
        )

        m = NeuralProphet(
            n_lags=0,
            n_forecasts=1,
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=True,
            seasonality_mode="additive",
            learning_rate=0.2,
            epochs=400,
            batch_size=64,
            loss_func="MSE",
        )

        reg_cols = [
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
        for col in reg_cols:
            m.add_future_regressor(col, normalize="minmax")

        df_train = df_b[["ds", "y"] + reg_cols].dropna().copy()
        m.fit(df_train, freq="h")

        np_path = MODEL_DIR / f"np_model_{station.pk}.np"
        np_save(m, str(np_path))
        print(f"[TRAIN] station {station.pk}: NP(y) сохранён в {np_path}")

        features_reg = reg_cols
        np_meta_path = MODEL_DIR / f"np_model_{station.pk}.meta.json"
        np_meta = {
            "station_id": station.pk,
            "cap_mw": cap_mw,
            "pr_for_expected": PR_FOR_EXPECTED,
            "features_reg": features_reg,
            "target": "y (MWh) = power_kw/1000",
            "note": "NP trained on direct y from SolarRecord",
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
