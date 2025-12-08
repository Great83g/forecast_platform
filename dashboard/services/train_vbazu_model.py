"""Training script for NeuralProphet + XGBoost using solar_hourly_kw.

This module adapts the standalone training flow to run inside the
forecast_platform repository and work against the project's database
schema. Models are saved into the shared ``models_cache`` directory so
other components can pick them up.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import xgboost as xgb
from neuralprophet import NeuralProphet, save

# === Paths ===
BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models_cache"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

NP_MODEL_FILE = MODEL_DIR / "trained_np_kw_8p8mw.np"
NP_META_FILE = MODEL_DIR / "trained_np_kw_8p8mw.meta.json"
XGB_MODEL_FILE = MODEL_DIR / "xgb_permw_kw_8p8mw.json"
XGB_META_FILE = MODEL_DIR / "xgb_permw_kw_8p8mw.meta.json"

# === Constants ===
CAP_MW = 8.8
PR_FOR_EXPECTED = 0.90
EXPECTED_UPPER = CAP_MW * 0.95  # физический верх

# === Postgres params (override via environment for local runs) ===
PG_HOST = os.getenv("PG_HOST", "127.0.0.1")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASS = os.getenv("PG_PASS", "Great83g")
PG_DB = os.getenv("PG_DB", "Solar10Mv")
PG_SCHEMA = os.getenv("PG_SCHEMA", "solar")
PG_TABLE = os.getenv("PG_TABLE", "solar_hourly_kw")  # ds, irradiation, air_temp, pv_temp, power_kw


# =========================
#  PG utils
# =========================
def pg_connect():
    return psycopg2.connect(
        dbname=PG_DB,
        host=PG_HOST,
        port=PG_PORT,
        user=PG_USER,
        password=PG_PASS,
    )


def load_history_from_db():
    """
    Load history from solar.solar_hourly_kw and compute target y (MWh).
    """

    con = pg_connect()
    try:
        sql = f"""
            SELECT
                ds,
                irradiation AS "Irradiation",
                air_temp    AS "Air_Temp",
                pv_temp     AS "PV_Temp",
                power_kw
            FROM "{PG_SCHEMA}"."{PG_TABLE}"
            WHERE power_kw IS NOT NULL
            ORDER BY ds;
        """
        df = pd.read_sql_query(sql, con, parse_dates=["ds"])
    finally:
        con.close()

    df["y"] = df["power_kw"] / 1000.0
    df = df.dropna(subset=["y"]).copy()
    return df


# =========================
#  Feature engineering
# =========================
def add_sun_geometry(df, ds_col: str = "ds", lat_deg: float = 47.86):
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


def add_common_features(df, ds_col: str = "ds"):
    df["hour"] = df[ds_col].dt.hour
    df["month"] = df[ds_col].dt.month
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)

    df["is_clear"] = ((df["Irradiation"] > 200) & (df["Air_Temp"] > 0)).astype(int)
    df["morning_peak_boost"] = (
        (df["hour"] == 6) & (df["Irradiation"] > 39)
    ).astype(int)
    df["evening_penalty"] = (
        (df["hour"] == 19) & (df["Irradiation"] > 39)
    ).astype(int)
    df["overdrive_flag"] = (
        (df["Irradiation"] > 950) & (df["Air_Temp"] > 30)
    ).astype(int)
    df["midday_penalty"] = (
        (df["hour"] >= 12) & (df["hour"] <= 14)
    ).astype(int)
    df["is_morning_active"] = (
        (df["hour"] == 6) & (df["Irradiation"] > 49)
    ).astype(int)

    df["PV_Temp"] = (
        df["Air_Temp"] + np.maximum(df["Irradiation"] - 50, 0) / 1000 * 20
    )

    df["y_expected"] = CAP_MW * (df["Irradiation"] / 1000) * PR_FOR_EXPECTED
    df["y_expected"] = df["y_expected"].clip(upper=EXPECTED_UPPER)
    df["y_expected_log"] = np.log1p(df["y_expected"] * 0.95)

    return df


# =========================
#  NeuralProphet
# =========================
def train_neuralprophet(df: pd.DataFrame):
    print("🔧 Обучение NeuralProphet (прямо на y)...")

    df = df.copy()
    df = add_common_features(df, "ds")
    df = add_sun_geometry(df, "ds", 47.86)

    df["dup_weight"] = 1
    df.loc[df["is_morning_active"] == 1, "dup_weight"] = 8
    df_dup = df.loc[df.index.repeat(df["dup_weight"])].copy()
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

    df_train = df_b[["ds", "y"] + features_reg].dropna().copy()

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
    for col in features_reg:
        model.add_future_regressor(col, normalize="minmax")

    model.fit(df_train, freq="h")
    save(model, str(NP_MODEL_FILE))

    NP_META_FILE.write_text(
        json.dumps(
            {
                "cap_mw": CAP_MW,
                "pr_for_expected": PR_FOR_EXPECTED,
                "features_reg": features_reg,
                "target": "y (MWh) = power_kw/1000",
                "note": "NP trained on direct y from solar_hourly_kw",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"✅ NeuralProphet сохранён: {NP_MODEL_FILE.name}, {NP_META_FILE.name}")


# =========================
#  XGBoost per-MW
# =========================
def train_xgb_permw(df: pd.DataFrame):
    print("🌲 Обучение XGBoost (per-MW)...")

    df = df.copy()
    df = add_common_features(df, "ds")
    df = add_sun_geometry(df, "ds", 47.86)

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

    df = df.dropna(subset=X_cols + ["y"]).copy()

    if len(df) == 0:
        raise RuntimeError("После фильтрации NaN нет данных для обучения XGBoost.")

    y_permw = df["y"] / CAP_MW

    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
    )

    model.fit(df[X_cols], y_permw)
    model.save_model(str(XGB_MODEL_FILE))

    XGB_META_FILE.write_text(
        json.dumps(
            {
                "cap_mw": CAP_MW,
                "X_cols": X_cols,
                "target": "y_per_MW = y / 8.8",
                "note": "XGB per-MW trained from solar_hourly_kw",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"✅ XGB per-MW сохранён: {XGB_MODEL_FILE.name}, {XGB_META_FILE.name}")


# =========================
#  MAIN
# =========================
def main():
    df = load_history_from_db()
    print(f"📊 Истории загружено строк: {len(df)}")

    df = df[df["y"] >= 0].copy()

    if len(df) == 0:
        raise RuntimeError("В истории нет строк с y >= 0 для обучения.")

    train_neuralprophet(df)
    train_xgb_permw(df)


if __name__ == "__main__":
    main()
