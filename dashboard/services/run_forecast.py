# dashboard/services/run_forecast.py

from __future__ import annotations
import numpy as np
import pandas as pd

from solar.models import SolarForecast
from dashboard.services.vc_weather import load_weather
from dashboard.services.forecast_engine import (
    load_models,
    make_features,
    heuristic,
    ensemble,
    run_forecast_for_station as _run_forecast_for_station,
)
from dashboard.services.train_models import train_models_for_station as _train_models_for_station


def run_forecast(station) -> int:
    """
    Запускает прогноз для одной станции (NP + XGB + эвристика + ансамбль).
    """
    # ----- 1. Загружаем модели для конкретной станции -----
    np_model, xgb_model = load_models(station)

    # ----- 2. Грузим прогноз погоды и делаем фичи -----
    df = load_weather(station.lat, station.lon)
    df = make_features(df)

    df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)

    if df.empty:
        return 0

    # ----- 3. NeuralProphet: только ds -----
    df_np = df[["ds"]].copy()
    np_pred_df = np_model.predict(df_np)
    pred_np = np_pred_df["yhat1"].values

    # ----- 4. XGBoost -----
    xgb_features = df[["Irradiation", "Air_Temp", "PV_Temp", "hour_sin", "hour_cos"]]
    pred_xgb = xgb_model.predict(xgb_features)

    # ----- 5. Эвристика -----
    pred_heur = heuristic(df)

    # ----- 6. Ансамбль -----
    pred_final = ensemble(pred_np, pred_xgb, pred_heur)

    # ----- 7. Сохранение в базу -----
    SolarForecast.objects.filter(station=station).delete()

    for i, row in df.iterrows():
        SolarForecast.objects.update_or_create(
            station=station,
            timestamp=row["ds"],
            defaults={
                "pred_np": float(pred_np[i]),
                "pred_xgb": float(pred_xgb[i]),
                "pred_heur": float(pred_heur[i]),
                "pred_final": float(pred_final[i]),
            },
        )

    return len(df)


# ==== Совместимость со старым импортом ====
# В некоторых местах код мог ожидать функции в этом модуле.
# Проксируем на новую реализацию из forecast_engine.
def run_forecast_for_station(station, days: int = 3) -> int:
    return _run_forecast_for_station(station, days=days)


# Сохраняем имя функции обучения моделей для старых импортов.
train_models_for_station = _train_models_for_station


__all__ = [
    "run_forecast",
    "run_forecast_for_station",
    "train_models_for_station",
]
