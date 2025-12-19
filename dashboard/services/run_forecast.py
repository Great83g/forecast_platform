"""dashboard/services/run_forecast.py

Тонкая обёртка над forecast_engine.
Нужна, чтобы views.py мог вызывать run_forecast_for_station(..., days=...).
"""

from __future__ import annotations

from .forecast_engine import run_forecast_for_station as _run_engine
from .forecast_engine import train_models_for_station

__all__ = ["run_forecast_for_station", "train_models_for_station"]


def run_forecast_for_station(station_id: int, days: int = 3):
    """Запуск прогноза на N дней вперёд."""
    return _run_engine(station_id, days=days)
