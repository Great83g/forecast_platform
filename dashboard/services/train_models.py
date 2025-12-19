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


def compute_cap_mw(df: pd.DataFrame) -> float:
    """
    Грубая оценка установленной мощности станции (MW) по истории.
    """
    if df.empty:
        return 1.0
    max_kw = float(df["Power_KW"].max())
    # минимальный кап 0.5 МВт
    return max(0.5, max_kw / 1000.0)


def add_features(df: pd.DataFrame, cap_mw: float) -> pd.DataFrame:
    """
    Добавляем все фичи, которые нужны и для XGB, и для NeuralProphet.

    На выходе есть:
      hour, month, hour_sin, month_sin,
      is_daylight, is_clear,
      morning_peak_boost, overdrive_flag, midday_penalty,
      Expected_MW, y_expected_log
    """
    df = df.copy()
    df["ds"] = pd.to_datetime(df["ds"])

    df["hour"] = df["ds"].dt.hour
    df["month"] = df["ds"].dt.month

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

    df["is_daylight"] = ((df["hour"] >= 6) & (df["hour"] <= 20)).astype(int)

    # Простой флаг "ясно"
    df["is_clear"] = ((df["Irradiation"] > 200) & (df["Air_Temp"] > -10)).astype(int)

    # Усиление утра (6:00, есть солнце)
    df["morning_peak_boost"] = (
        (df["hour"] == 6) & (df["Irradiation"] > 39)
    ).astype(int)

    # Лёгкий штраф середины дня, когда часто клип
    df["midday_penalty"] = ((df["hour"] >= 12) & (df["hour"] <= 14)).astype(int)

    # Overdrive при очень высокой радиации и температуре
    df["overdrive_flag"] = (
        (df["Irradiation"] > 900) & (df["Air_Temp"] > 25)
    ).astype(int)

    # Простая эвристика expected MW (по PR)
    df["Expected_MW"] = (
        cap_mw * (df["Irradiation"] / 1000.0) * PR_FOR_EXPECTED
    ).clip(0, cap_mw * 0.95)

    df["y_expected_log"] = np.log1p(df["Expected_MW"])

    return df


def train_models_for_station(station) -> Tuple[int, Path | None, Path | None]:
    """
    Обучаем XGB(per-MW) и NeuralProphet(y_mw) для одной станции.

    Сохраняем:
      - XGB:  xgb_model_{pk}.json + .meta.json
      - NP:   np_model_{pk}.np   + .meta.json

    Возвращаем: (кол-во строк истории, путь к NP, путь к XGB)
    """
    df = get_history_dataframe(station)
    if df.empty:
        return 0, None, None

    cap_mw = compute_cap_mw(df)
    print(f"[TRAIN] station {station.pk}: оценка cap_mw={cap_mw:.3f}")

    # таргет в MW
    df["y_mw"] = (df["Power_KW"] / 1000.0).clip(lower=0)

    # общее фичеобразование
    df = add_features(df, cap_mw)
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
        "is_daylight",
        "is_clear",
        "morning_peak_boost",
        "overdrive_flag",
        "midday_penalty",
    ]

    df["y_permw"] = (df["y_mw"] / cap_mw).clip(lower=0)
    df_xgb = df.dropna(subset=X_cols + ["y_permw"]).copy()

    xgb_path: Path | None = None
    if len(df_xgb) < 50:
        print(
            f"[TRAIN] station {station.pk}: недостаточно строк для XGB "
            f"({len(df_xgb)} < 50)"
        )
    else:
        try:
            model_xgb = xgb.XGBRegressor(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="reg:squarederror",
                tree_method="hist",
            )
            model_xgb.fit(df_xgb[X_cols], df_xgb["y_permw"])

            xgb_path = MODEL_DIR / f"xgb_model_{station.pk}.json"
            model_xgb.save_model(str(xgb_path))

            meta_xgb = {
                "station_id": station.pk,
                "X_cols": X_cols,
                "cap_mw_used": cap_mw,
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
        df_np = df_np[
            [
                "ds",
                "y_mw",
                "Irradiation",
                "Air_Temp",
                "PV_Temp",
                "hour_sin",
                "month_sin",
                "is_daylight",
                "is_clear",
                "y_expected_log",
                "morning_peak_boost",
                "overdrive_flag",
                "midday_penalty",
            ]
        ].dropna()

        df_np.rename(columns={"y_mw": "y"}, inplace=True)

        print(
            f"[TRAIN] station {station.pk}: старт обучения NP(y_mw), строк={len(df_np)}"
        )

        m = NeuralProphet(
            n_lags=0,
            n_forecasts=1,
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            learning_rate=0.5,
            epochs=300,
            batch_size=64,
            loss_func="Huber",
        )

        reg_cols = [
            "Irradiation",
            "Air_Temp",
            "PV_Temp",
            "hour_sin",
            "month_sin",
            "is_daylight",
            "is_clear",
            "y_expected_log",
            "morning_peak_boost",
            "overdrive_flag",
            "midday_penalty",
        ]
        for col in reg_cols:
            m.add_future_regressor(col, normalize="minmax")

        m.fit(df_np, freq="H")

        np_path = MODEL_DIR / f"np_model_{station.pk}.np"
        np_save(m, str(np_path))
        print(f"[TRAIN] station {station.pk}: NP(y_mw) сохранён в {np_path}")

        features_reg = reg_cols
        np_meta_path = MODEL_DIR / f"np_model_{station.pk}.meta.json"
        np_meta = {
            "station_id": station.pk,
            "features_reg": features_reg,
            "cap_mw_used": cap_mw,
            "note": "NP trained on direct y_mw from SolarRecord",
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
