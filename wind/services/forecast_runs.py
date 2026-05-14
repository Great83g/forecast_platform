from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass
from datetime import date
from typing import Iterable

import pandas as pd
from django.db import transaction
from django.utils import timezone

from stations.models import Station
from wind.models import WindForecast, WindForecastRun

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WindForecastPayload:
    timestamp: object
    pred_heur: float | None
    pred_final: float | None
    weather_source: str
    air_temp_fc: float | None = None
    wind_speed_fc: float | None = None
    wind_direction_fc: float | None = None
    cloudcover_fc: float | None = None
    humidity_fc: float | None = None
    precip_fc: float | None = None


def _to_datetime(value):
    return value.to_pydatetime() if hasattr(value, "to_pydatetime") else value


def _target_bounds(rows: list[WindForecastPayload]):
    timestamps = [_to_datetime(row.timestamp) for row in rows if row.timestamp is not None]
    if not timestamps:
        return None, None
    return min(timestamps), max(timestamps)


def create_wind_forecast_run(
    *,
    station: Station,
    forecast_scope: str,
    forecast_base_date: date,
    provider: str,
    horizon_days: int,
    rows: Iterable[WindForecastPayload],
) -> WindForecastRun:
    payloads = list(rows)
    min_target, max_target = _target_bounds(payloads)

    with transaction.atomic():
        run = WindForecastRun.objects.create(
            station=station,
            forecast_scope=forecast_scope,
            forecast_base_date=forecast_base_date,
            provider=provider or "",
            horizon_days=max(int(horizon_days or 1), 1),
        )
        logger.info(
            "Saving wind forecast station_id=%s run_id=%s forecast_base_date=%s rows_count=%s min_target_datetime=%s max_target_datetime=%s",
            station.pk,
            run.pk,
            forecast_base_date,
            len(payloads),
            min_target,
            max_target,
        )
        WindForecast.objects.bulk_create(
            [
                WindForecast(
                    station=station,
                    forecast_run=run,
                    forecast_scope=forecast_scope,
                    timestamp=_to_datetime(row.timestamp),
                    pred_heur=row.pred_heur,
                    pred_final=row.pred_final,
                    weather_source=row.weather_source or provider or "",
                    air_temp_fc=row.air_temp_fc,
                    wind_speed_fc=row.wind_speed_fc,
                    wind_direction_fc=row.wind_direction_fc,
                    cloudcover_fc=row.cloudcover_fc,
                    humidity_fc=row.humidity_fc,
                    precip_fc=row.precip_fc,
                )
                for row in payloads
            ],
            batch_size=1000,
        )
    return run


def latest_wind_forecast_rows(queryset) -> list[WindForecast]:
    rows = sorted(
        list(queryset),
        key=lambda row: (
            row.timestamp,
            timezone.localtime(row.forecast_run.created_at) if row.forecast_run_id else timezone.localtime(row.created_at),
            row.forecast_run_id or 0,
            row.pk or 0,
        ),
        reverse=True,
    )
    latest_by_timestamp: OrderedDict[object, WindForecast] = OrderedDict()
    for row in rows:
        latest_by_timestamp.setdefault(row.timestamp, row)
    return sorted(latest_by_timestamp.values(), key=lambda row: row.timestamp)


def dataframe_from_latest_wind_forecasts(queryset) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "timestamp": row.timestamp,
                "pred_heur": row.pred_heur,
                "pred_final": row.pred_final,
                "wind_speed_fc": row.wind_speed_fc,
                "air_temp_fc": row.air_temp_fc,
                "cloudcover_fc": row.cloudcover_fc,
                "humidity_fc": row.humidity_fc,
                "precip_fc": row.precip_fc,
                "weather_source": row.weather_source,
            }
            for row in latest_wind_forecast_rows(queryset)
        ]
    )
