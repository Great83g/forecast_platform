from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional
from urllib.parse import urlencode

from django.db.models import Avg
from django.utils import timezone

from solar.models import SolarForecast, SolarRecord
from stations.models import Station


@dataclass(frozen=True)
class EnergySummary:
    station_id: int
    station_name: str
    date: date
    energy_kwh: float
    points_count: int


@dataclass(frozen=True)
class PlanFactSummary:
    station_id: int
    station_name: str
    date: date
    fact_kwh: float
    plan_kwh: float
    deviation_kwh: float
    deviation_percent: Optional[float]
    mape_percent: Optional[float]
    points_count: int


def _today() -> date:
    return timezone.localdate()


def _forecast_value_to_kw(value: Optional[float], station_capacity_mw: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        return None

    capacity = None
    try:
        if station_capacity_mw is not None:
            capacity = float(station_capacity_mw)
    except (TypeError, ValueError):
        capacity = None

    # Legacy forecasts in this project could be stored in MW, while current rows use kW.
    mw_threshold = max((capacity or 0.0) * 2.0, 10.0)
    if abs(normalized) <= mw_threshold:
        return normalized * 1000.0
    return normalized


def _forecast_plan_value(row: dict, station_capacity_mw: Optional[float]) -> Optional[float]:
    for field in ("pred_final", "pred_heur"):
        value = _forecast_value_to_kw(row.get(field), station_capacity_mw)
        if value is not None and (field != "pred_final" or value > 0):
            return value
    return None


def _station(station_id: int) -> Station:
    return Station.objects.get(pk=station_id)


def _sum_generation_for_date(station: Station, target_date: date) -> EnergySummary:
    rows = (
        SolarRecord.objects.filter(
            station=station,
            history_scope=SolarRecord.HISTORY_SCOPE_MAIN,
            timestamp__date=target_date,
        )
        .values("timestamp")
        .annotate(power_kw=Avg("power_kw"))
        .order_by("timestamp")
    )
    values = [float(row["power_kw"]) for row in rows if row.get("power_kw") is not None]
    return EnergySummary(
        station_id=station.pk,
        station_name=station.name,
        date=target_date,
        energy_kwh=sum(values),
        points_count=len(values),
    )


def _sum_forecast_for_date(station: Station, target_date: date) -> EnergySummary:
    rows = (
        SolarForecast.objects.filter(
            station=station,
            forecast_scope=SolarForecast.SCOPE_MAIN,
            timestamp__date=target_date,
        )
        .values("timestamp")
        .annotate(pred_final=Avg("pred_final"), pred_heur=Avg("pred_heur"))
        .order_by("timestamp")
    )
    values = []
    for row in rows:
        value = _forecast_plan_value(row, station.capacity_mw)
        if value is not None:
            values.append(value)
    return EnergySummary(
        station_id=station.pk,
        station_name=station.name,
        date=target_date,
        energy_kwh=sum(values),
        points_count=len(values),
    )


def _planfact_for_date(station: Station, target_date: date) -> PlanFactSummary:
    fact = _sum_generation_for_date(station, target_date)
    plan = _sum_forecast_for_date(station, target_date)

    deviation = fact.energy_kwh - plan.energy_kwh
    deviation_percent = (deviation / plan.energy_kwh * 100.0) if plan.energy_kwh else None

    mape_values = []
    history_rows = (
        SolarRecord.objects.filter(
            station=station,
            history_scope=SolarRecord.HISTORY_SCOPE_MAIN,
            timestamp__date=target_date,
        )
        .values("timestamp")
        .annotate(power_kw=Avg("power_kw"))
    )
    forecast_rows = (
        SolarForecast.objects.filter(
            station=station,
            forecast_scope=SolarForecast.SCOPE_MAIN,
            timestamp__date=target_date,
        )
        .values("timestamp")
        .annotate(pred_final=Avg("pred_final"), pred_heur=Avg("pred_heur"))
    )
    fact_map = {row["timestamp"]: float(row["power_kw"]) for row in history_rows if row.get("power_kw") is not None}
    plan_map = {}
    for row in forecast_rows:
        value = _forecast_plan_value(row, station.capacity_mw)
        if value is not None:
            plan_map[row["timestamp"]] = value

    fact_values = [value for value in fact_map.values() if value is not None]
    peak_fact_kw = max(fact_values) if fact_values else 0.0
    min_fact_for_mape_kw = max(1.0, peak_fact_kw * 0.10)
    for timestamp, fact_kw in fact_map.items():
        plan_kw = plan_map.get(timestamp)
        if plan_kw is None or fact_kw <= 0 or fact_kw < min_fact_for_mape_kw:
            continue
        mape_values.append(abs((fact_kw - plan_kw) / fact_kw) * 100.0)

    return PlanFactSummary(
        station_id=station.pk,
        station_name=station.name,
        date=target_date,
        fact_kwh=fact.energy_kwh,
        plan_kwh=plan.energy_kwh,
        deviation_kwh=deviation,
        deviation_percent=deviation_percent,
        mape_percent=(sum(mape_values) / len(mape_values)) if mape_values else None,
        points_count=max(fact.points_count, plan.points_count),
    )


def get_yesterday_generation(station_id: int) -> EnergySummary:
    station = _station(station_id)
    return _sum_generation_for_date(station, _today() - timedelta(days=1))


def get_tomorrow_forecast(station_id: int) -> EnergySummary:
    station = _station(station_id)
    return _sum_forecast_for_date(station, _today() + timedelta(days=1))


def get_today_planfact(station_id: int) -> PlanFactSummary:
    station = _station(station_id)
    return _planfact_for_date(station, _today())


def get_yesterday_planfact(station_id: int) -> PlanFactSummary:
    station = _station(station_id)
    return _planfact_for_date(station, _today() - timedelta(days=1))


def build_navigation_action(intent: str, station_id: int):
    station = _station(station_id)
    today = _today()
    if intent == "open_tomorrow_forecast":
        target_date = today + timedelta(days=1)
        query = urlencode({"date": target_date.isoformat(), "scope": SolarForecast.SCOPE_MAIN})
        return {"type": "navigate", "url": f"/dashboard/station/{station.pk}/forecast/list/?{query}"}
    if intent == "open_planfact_today":
        target_date = today
        query = urlencode({"date_from": target_date.isoformat(), "date_to": target_date.isoformat()})
        return {"type": "navigate", "url": f"/dashboard/station/{station.pk}/?{query}"}
    if intent == "open_planfact_yesterday":
        target_date = today - timedelta(days=1)
        query = urlencode({"date_from": target_date.isoformat(), "date_to": target_date.isoformat()})
        return {"type": "navigate", "url": f"/dashboard/station/{station.pk}/?{query}"}
    return None
