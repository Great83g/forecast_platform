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


def _as_float(value, default=0.0):
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _decimal_mw(value):
    if value is None:
        return None
    return Decimal(str(round(float(value), 4)))


def _decimal_percent(value):
    if value is None:
        return None
    return Decimal(str(round(float(value), 3)))


def _build_ess_calculator(ess_config):
    if ess_config is None:
        return None

    capacity_mwh = _as_float(ess_config.capacity_mwh)
    power_mw = _as_float(ess_config.power_mw)
    pcs_power_mw = _as_float(ess_config.pcs_power_mw)
    charge_efficiency = _as_float(ess_config.charge_efficiency)
    discharge_efficiency = _as_float(ess_config.discharge_efficiency)
    timestep_minutes = _as_float(ess_config.timestep_minutes, 60.0)

    if (
        capacity_mwh <= 0
        or power_mw <= 0
        or pcs_power_mw <= 0
        or charge_efficiency <= 0
        or discharge_efficiency <= 0
        or timestep_minutes <= 0
    ):
        return None

    usable_power_mw = min(power_mw, pcs_power_mw)
    step_hours = timestep_minutes / 60.0
    soc_mwh = capacity_mwh * _as_float(ess_config.soc_initial_percent) / 100.0
    soc_min_mwh = capacity_mwh * _as_float(ess_config.soc_min_percent) / 100.0
    soc_max_mwh = capacity_mwh * _as_float(ess_config.soc_max_percent) / 100.0
    soc_mwh = min(max(soc_mwh, soc_min_mwh), soc_max_mwh)

    def calculate(plan_mw, fact_mw):
        nonlocal soc_mwh

        if plan_mw is None or fact_mw is None:
            return {}

        deviation_mw = plan_mw - fact_mw
        ess_charge_mw = 0.0
        ess_discharge_mw = 0.0
        ess_command_mw = 0.0

        if deviation_mw > 0:
            requested_discharge_mw = deviation_mw
            available_discharge_mwh = max(0.0, soc_mwh - soc_min_mwh)
            # Ограничиваем по энергии с учетом КПД, чтобы SOC не ушел ниже soc_min_mwh.
            max_discharge_mw_by_energy = available_discharge_mwh * discharge_efficiency / step_hours
            ess_discharge_mw = max(
                0.0,
                min(requested_discharge_mw, usable_power_mw, max_discharge_mw_by_energy),
            )
            soc_mwh -= ess_discharge_mw * step_hours / discharge_efficiency
            ess_command_mw = ess_discharge_mw
        elif deviation_mw < 0:
            requested_charge_mw = abs(deviation_mw)
            available_charge_mwh = max(0.0, soc_max_mwh - soc_mwh)
            # Ограничиваем по энергии с учетом КПД, чтобы SOC не превысил soc_max_mwh.
            max_charge_mw_by_energy = available_charge_mwh / (step_hours * charge_efficiency)
            ess_charge_mw = max(
                0.0,
                min(requested_charge_mw, usable_power_mw, max_charge_mw_by_energy),
            )
            soc_mwh += ess_charge_mw * step_hours * charge_efficiency
            ess_command_mw = -ess_charge_mw

        soc_mwh = min(max(soc_mwh, soc_min_mwh), soc_max_mwh)
        output_after_ess_mw = fact_mw + ess_discharge_mw - ess_charge_mw
        unbalanced_mw = plan_mw - output_after_ess_mw
        soc_percent = soc_mwh / capacity_mwh * 100.0

        return {
            "ess_command_mw": ess_command_mw,
            "ess_charge_mw": ess_charge_mw,
            "ess_discharge_mw": ess_discharge_mw,
            "soc_percent": soc_percent,
            "soc_mwh": soc_mwh,
            "output_after_ess_mw": output_after_ess_mw,
            "unbalanced_mw": unbalanced_mw,
        }

    return calculate


def populate_simulation_points(run: ESSSimulationRun) -> dict[str, int]:
    """
    Stage 3 Virtual ESS simulation.

    Loads station forecast and actual history by timestamp, computes deviation, then
    applies a simple hourly virtual ESS charge/discharge model. This does not touch
    the existing forecast/training/tracker pipeline.
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
    ess_calculator = _build_ess_calculator(run.ess_config)

    timestamps = sorted(set(forecast_by_timestamp) | set(actual_by_timestamp))
    points = []
    for ts in timestamps:
        forecast_mw = forecast_by_timestamp.get(ts)
        actual_mw = actual_by_timestamp.get(ts)
        deviation_mw = None
        if forecast_mw is not None and actual_mw is not None:
            deviation_mw = forecast_mw - actual_mw

        ess_values = ess_calculator(forecast_mw, actual_mw) if ess_calculator else {}
        points.append(
            ESSSimulationPoint(
                run=run,
                timestamp=ts,
                plan_mw=_decimal_mw(forecast_mw),
                fact_mw=_decimal_mw(actual_mw),
                deviation_mw=_decimal_mw(deviation_mw),
                ess_command_mw=_decimal_mw(ess_values.get("ess_command_mw")),
                ess_charge_mw=_decimal_mw(ess_values.get("ess_charge_mw")),
                ess_discharge_mw=_decimal_mw(ess_values.get("ess_discharge_mw")),
                soc_percent=_decimal_percent(ess_values.get("soc_percent")),
                soc_mwh=_decimal_mw(ess_values.get("soc_mwh")),
                output_after_ess_mw=_decimal_mw(ess_values.get("output_after_ess_mw")),
                unbalanced_mw=_decimal_mw(ess_values.get("unbalanced_mw")),
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


def build_run_summary(run: ESSSimulationRun) -> dict[str, float | None]:
    points = list(run.points.all())
    step_hours = _as_float(getattr(run.ess_config, "timestep_minutes", None), 60.0) / 60.0
    if step_hours <= 0:
        step_hours = 1.0

    def values(field_name):
        return [_as_float(getattr(point, field_name), None) for point in points if getattr(point, field_name) is not None]

    plan_values = values("plan_mw")
    fact_values = values("fact_mw")
    after_values = values("output_after_ess_mw")
    deviation_values = values("deviation_mw")
    unbalanced_values = values("unbalanced_mw")
    soc_values = values("soc_percent")
    charge_values = values("ess_charge_mw")
    discharge_values = values("ess_discharge_mw")

    before_abs = sum(abs(value) * step_hours for value in deviation_values)
    after_abs = sum(abs(value) * step_hours for value in unbalanced_values)
    improvement_percent = None
    if before_abs > 0:
        improvement_percent = (before_abs - after_abs) / before_abs * 100.0

    return {
        "plan_energy_mwh": sum(plan_values) * step_hours,
        "fact_energy_mwh": sum(fact_values) * step_hours,
        "after_ess_energy_mwh": sum(after_values) * step_hours if after_values else None,
        "abs_deviation_before_mwh": before_abs,
        "abs_deviation_after_mwh": after_abs if unbalanced_values else None,
        "improvement_percent": improvement_percent,
        "min_soc_percent": min(soc_values) if soc_values else None,
        "max_soc_percent": max(soc_values) if soc_values else None,
        "total_charge_mwh": sum(charge_values) * step_hours,
        "total_discharge_mwh": sum(discharge_values) * step_hours,
    }
