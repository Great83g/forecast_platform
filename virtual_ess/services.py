from datetime import datetime, time
from decimal import Decimal

from django.db import transaction
from django.db.models import Avg
from django.utils import timezone

from solar.models import SolarForecast, SolarRecord

from .models import ESSSimulationPoint, ESSSimulationRun


def _day_start(value):
    dt = datetime.combine(value, time.min)
    if timezone.is_naive(dt):
        return timezone.make_aware(dt, timezone.get_current_timezone())
    return dt


def _day_end(value):
    dt = datetime.combine(value, time.max)
    if timezone.is_naive(dt):
        return timezone.make_aware(dt, timezone.get_current_timezone())
    return dt


def _forecast_value_kw(pred_final, pred_heur):
    if pred_final is not None:
        try:
            value = float(pred_final)
        except (TypeError, ValueError):
            value = None
        if value is not None and value > 0:
            return value

    if pred_heur is not None:
        try:
            return float(pred_heur)
        except (TypeError, ValueError):
            return None
    return None


def _kw_to_mw(value_kw):
    if value_kw is None:
        return None
    try:
        return float(value_kw) / 1000.0
    except (TypeError, ValueError):
        return None


def _decimal_mw(value):
    if value is None:
        return None
    return Decimal(str(round(float(value), 4)))


def populate_simulation_points(run: ESSSimulationRun) -> dict[str, int]:
    """
    Stage 2 data loader: joins station forecast and actual history by timestamp.

    No ESS command, SOC, charge/discharge or optimization logic is calculated here.
    """
    start_at = _day_start(run.date_from)
    end_at = _day_end(run.date_to)

    forecast_rows = (
        SolarForecast.objects.filter(
            station=run.station,
            forecast_scope=SolarForecast.SCOPE_MAIN,
            timestamp__gte=start_at,
            timestamp__lte=end_at,
        )
        .values("timestamp")
        .annotate(pred_final=Avg("pred_final"), pred_heur=Avg("pred_heur"))
        .order_by("timestamp")
    )
    actual_rows = (
        SolarRecord.objects.filter(
            station=run.station,
            history_scope=SolarRecord.HISTORY_SCOPE_MAIN,
            timestamp__gte=start_at,
            timestamp__lte=end_at,
        )
        .values("timestamp")
        .annotate(power_kw=Avg("power_kw"))
        .order_by("timestamp")
    )

    forecast_by_timestamp = {
        row["timestamp"]: _kw_to_mw(_forecast_value_kw(row.get("pred_final"), row.get("pred_heur")))
        for row in forecast_rows
    }
    actual_by_timestamp = {row["timestamp"]: _kw_to_mw(row.get("power_kw")) for row in actual_rows}

    timestamps = sorted(set(forecast_by_timestamp) | set(actual_by_timestamp))
    points = []
    for ts in timestamps:
        forecast_mw = forecast_by_timestamp.get(ts)
        actual_mw = actual_by_timestamp.get(ts)
        deviation_mw = None
        if forecast_mw is not None and actual_mw is not None:
            deviation_mw = forecast_mw - actual_mw

        points.append(
            ESSSimulationPoint(
                run=run,
                timestamp=ts,
                plan_mw=_decimal_mw(forecast_mw),
                fact_mw=_decimal_mw(actual_mw),
                deviation_mw=_decimal_mw(deviation_mw),
            )
        )

    with transaction.atomic():
        run.points.all().delete()
        if points:
            ESSSimulationPoint.objects.bulk_create(points, batch_size=1000)
        run.status = ESSSimulationRun.STATUS_FINISHED
        run.finished_at = timezone.now()
        run.save(update_fields=["status", "finished_at"])

    return {
        "forecast_rows": len(forecast_by_timestamp),
        "actual_rows": len(actual_by_timestamp),
        "points": len(points),
    }
