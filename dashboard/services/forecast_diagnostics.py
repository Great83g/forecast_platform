from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Iterable, Optional

import pandas as pd
from django.utils import timezone

from solar.models import SolarForecast
from stations.models import Station

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ForecastModeComparisonRow:
    timestamp: object
    forecast_weather_irr: Optional[float]
    postfact_weather_irr: Optional[float]
    forecast_raw_mw: Optional[float]
    postfact_raw_mw: Optional[float]
    forecast_final_mw: Optional[float]
    postfact_final_mw: Optional[float]


def _kw_to_mw(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not pd.notna(value):
        return None
    return value / 1000.0


def _relative_diff_percent(left: Optional[float], right: Optional[float]) -> Optional[float]:
    if left is None or right is None:
        return None
    denom = max(abs(float(left)), abs(float(right)), 1e-6)
    return abs(float(left) - float(right)) / denom * 100.0


def compare_forecast_modes(
    station: Station,
    date_from: date,
    date_to: date,
    *,
    forecast_scope: str = SolarForecast.SCOPE_MAIN,
    postfact_scope: str = SolarForecast.SCOPE_TEST,
    threshold_percent: float = 5.0,
) -> list[ForecastModeComparisonRow]:
    """Compare saved operational and postfact forecasts hour by hour.

    The comparison focuses on columns needed to diagnose tracker stations:
    weather irradiation, raw final before guardrails, and final power after all
    station calibration/tracker/clipping steps. Any relative difference above
    ``threshold_percent`` is logged with enough context for production logs.
    """
    tz = timezone.get_current_timezone()
    start = timezone.datetime.combine(date_from, timezone.datetime.min.time()).replace(tzinfo=tz)
    end = timezone.datetime.combine(date_to, timezone.datetime.max.time()).replace(tzinfo=tz)

    qs = SolarForecast.objects.filter(
        station=station,
        timestamp__gte=start,
        timestamp__lte=end,
        forecast_scope__in=[forecast_scope, postfact_scope],
    ).order_by("timestamp", "forecast_scope")

    by_scope: dict[str, dict[object, SolarForecast]] = {forecast_scope: {}, postfact_scope: {}}
    for row in qs:
        by_scope.setdefault(row.forecast_scope, {})[row.timestamp] = row

    timestamps = sorted(set(by_scope.get(forecast_scope, {})) | set(by_scope.get(postfact_scope, {})))
    rows: list[ForecastModeComparisonRow] = []
    for ts in timestamps:
        forecast = by_scope.get(forecast_scope, {}).get(ts)
        postfact = by_scope.get(postfact_scope, {}).get(ts)
        item = ForecastModeComparisonRow(
            timestamp=ts,
            forecast_weather_irr=float(forecast.irradiation_fc) if forecast and forecast.irradiation_fc is not None else None,
            postfact_weather_irr=float(postfact.irradiation_fc) if postfact and postfact.irradiation_fc is not None else None,
            forecast_raw_mw=_kw_to_mw(forecast.pred_final_raw if forecast else None),
            postfact_raw_mw=_kw_to_mw(postfact.pred_final_raw if postfact else None),
            forecast_final_mw=_kw_to_mw(forecast.pred_final if forecast else None),
            postfact_final_mw=_kw_to_mw(postfact.pred_final if postfact else None),
        )
        rows.append(item)

        diffs = {
            "weather_irr": _relative_diff_percent(item.forecast_weather_irr, item.postfact_weather_irr),
            "raw_mw": _relative_diff_percent(item.forecast_raw_mw, item.postfact_raw_mw),
            "final_mw": _relative_diff_percent(item.forecast_final_mw, item.postfact_final_mw),
        }
        exceeded = {name: diff for name, diff in diffs.items() if diff is not None and diff > threshold_percent}
        if exceeded:
            logger.warning(
                "[FORECAST_COMPARE] station=%s ts=%s forecast_scope=%s postfact_scope=%s diffs=%s "
                "forecast_weather_irr=%s postfact_weather_irr=%s forecast_raw_mw=%s postfact_raw_mw=%s "
                "forecast_final_mw=%s postfact_final_mw=%s",
                station.pk,
                ts,
                forecast_scope,
                postfact_scope,
                {key: round(value, 2) for key, value in exceeded.items()},
                item.forecast_weather_irr,
                item.postfact_weather_irr,
                item.forecast_raw_mw,
                item.postfact_raw_mw,
                item.forecast_final_mw,
                item.postfact_final_mw,
            )
    return rows


def comparison_rows_to_dataframe(rows: Iterable[ForecastModeComparisonRow]) -> pd.DataFrame:
    return pd.DataFrame([row.__dict__ for row in rows])
