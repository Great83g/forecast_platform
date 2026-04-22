from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

PANEL_MODEL = "SP-N16/144HG"
PANEL_POWER_W = 580
PANEL_AREA_M2 = 2.58
PANEL_AREA_WITH_GAPS_M2 = 3.1
PANEL_WEIGHT_KG = 32
DEFAULT_SPECIFIC_YIELD = 1450
DEFAULT_COST_PER_KW = 550_000

CALC_MODES = [
    "consumption",
    "appliances",
    "budget",
    "roof_area",
    "max_roof",
    "utility_power",
    "utility_land",
]

ROOF_COEFFICIENTS = {
    "flat": 0.8,
    "simple": 0.7,
    "complex": 0.5,
}

SHADING_COEFFICIENTS = {
    "none": 1.0,
    "low": 0.9,
    "medium": 0.8,
    "high": 0.65,
}


@dataclass
class BaseSystemResult:
    system_kw: float
    panel_count: int
    panel_model: str
    area_required: float
    area_panels: float
    weight: float
    inverter_size: float
    annual_generation: float
    monthly_generation: float
    coverage_percent: float
    estimated_cost: float
    payback_years: float | None
    roof_fit: str | None = None


def _round2(value: float) -> float:
    return round(float(value), 2)


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _build_base_result(*, system_kw: float, annual_kwh_need: float = 0.0, tariff: float = 0.0, yield_factor: float = DEFAULT_SPECIFIC_YIELD, roof_area: float | None = None) -> BaseSystemResult:
    system_kw = max(system_kw, 0.0)
    panel_count = int(math.ceil(system_kw * 1000 / PANEL_POWER_W)) if system_kw else 0
    area_panels = panel_count * PANEL_AREA_M2
    area_required = panel_count * PANEL_AREA_WITH_GAPS_M2
    weight = panel_count * PANEL_WEIGHT_KG
    annual_generation = system_kw * yield_factor
    monthly_generation = annual_generation / 12 if annual_generation else 0
    coverage_percent = 0.0
    if annual_kwh_need > 0:
        coverage_percent = min(annual_generation / annual_kwh_need * 100, 100.0)

    estimated_cost = system_kw * DEFAULT_COST_PER_KW
    yearly_savings = annual_generation * max(tariff, 0)
    payback_years = None
    if yearly_savings > 0:
        payback_years = estimated_cost / yearly_savings

    roof_fit = None
    if roof_area is not None and roof_area > 0:
        if area_required <= roof_area:
            roof_fit = "true"
        elif area_panels <= roof_area:
            roof_fit = "partial"
        else:
            roof_fit = "false"

    return BaseSystemResult(
        system_kw=_round2(system_kw),
        panel_count=panel_count,
        panel_model=PANEL_MODEL,
        area_required=_round2(area_required),
        area_panels=_round2(area_panels),
        weight=_round2(weight),
        inverter_size=_round2(system_kw * 0.8),
        annual_generation=_round2(annual_generation),
        monthly_generation=_round2(monthly_generation),
        coverage_percent=_round2(coverage_percent),
        estimated_cost=_round2(estimated_cost),
        payback_years=_round2(payback_years) if payback_years is not None else None,
        roof_fit=roof_fit,
    )


def calculate(mode: str, inputs: dict[str, Any]) -> dict[str, Any]:
    if mode not in CALC_MODES:
        raise ValueError(f"Unsupported mode: {mode}")

    yield_factor = _to_float(inputs.get("specific_yield"), DEFAULT_SPECIFIC_YIELD)
    tariff = _to_float(inputs.get("tariff"), 0)

    if mode == "consumption":
        monthly_kwh = _to_float(inputs.get("monthly_kwh"), 0)
        annual_kwh = monthly_kwh * 12
        system_kw = annual_kwh / yield_factor if yield_factor > 0 else 0
        result = _build_base_result(system_kw=system_kw, annual_kwh_need=annual_kwh, tariff=tariff, yield_factor=yield_factor)
    elif mode == "appliances":
        from .appliance_calculator import calculate_from_appliances

        app_calc = calculate_from_appliances(inputs.get("appliances") or [])
        annual_kwh = app_calc["monthly_kwh"] * 12
        system_kw = annual_kwh / yield_factor if yield_factor > 0 else 0
        result = _build_base_result(system_kw=system_kw, annual_kwh_need=annual_kwh, tariff=tariff, yield_factor=yield_factor)
        result_dict = result.__dict__
        result_dict.update(app_calc)
        result = BaseSystemResult(**{k: result_dict[k] for k in BaseSystemResult.__dataclass_fields__.keys()})
    elif mode == "budget":
        budget = _to_float(inputs.get("budget"), 0)
        cost_per_kw = _to_float(inputs.get("cost_per_kw"), DEFAULT_COST_PER_KW)
        system_kw = budget / cost_per_kw if cost_per_kw > 0 else 0
        result = _build_base_result(system_kw=system_kw, tariff=tariff, yield_factor=yield_factor)
    elif mode == "roof_area":
        from .roof_calculator import calculate_roof_capacity_kw

        roof_area = _to_float(inputs.get("roof_area"), 0)
        roof_type = inputs.get("roof_type", "simple")
        shading = inputs.get("shading", "none")
        system_kw = calculate_roof_capacity_kw(roof_area=roof_area, roof_type=roof_type, shading=shading)
        result = _build_base_result(system_kw=system_kw, tariff=tariff, yield_factor=yield_factor, roof_area=roof_area)
    elif mode == "max_roof":
        roof_area = _to_float(inputs.get("roof_area"), 0)
        panel_count = math.floor(roof_area / PANEL_AREA_WITH_GAPS_M2) if roof_area > 0 else 0
        system_kw = panel_count * PANEL_POWER_W / 1000
        result = _build_base_result(system_kw=system_kw, tariff=tariff, yield_factor=yield_factor, roof_area=roof_area)
    elif mode == "utility_power":
        from .utility_calculator import calculate_by_target_power

        return calculate_by_target_power(_to_float(inputs.get("target_mw"), 0), yield_factor)
    else:  # utility_land
        from .utility_calculator import calculate_by_land

        return calculate_by_land(_to_float(inputs.get("land_hectares"), 0), yield_factor)

    variants = _build_variants(result, yield_factor, tariff)

    battery = None
    if bool(inputs.get("need_battery")):
        backup_hours = _to_float(inputs.get("backup_hours"), 4)
        critical_load_kw = _to_float(inputs.get("critical_load_kw"), max(result.system_kw * 0.25, 1.0))
        battery = {
            "battery_kwh": _round2(critical_load_kw * backup_hours),
            "backup_hours": _round2(backup_hours),
            "critical_load_kw": _round2(critical_load_kw),
        }

    result_payload = dict(result.__dict__)
    yearly_bill_without = annual_need_cost = _round2(result.annual_generation * 0 if False else 0)
    if tariff > 0:
        # Approximate comparison based on calculated coverage/consumption profile.
        annual_need_kwh = (result.annual_generation * 100 / result.coverage_percent) if result.coverage_percent > 0 else result.annual_generation
        yearly_bill_without = _round2(annual_need_kwh * tariff)
        yearly_bill_with = _round2(max(annual_need_kwh - result.annual_generation, 0) * tariff)
        result_payload["comparison"] = {
            "yearly_bill_without_spp": yearly_bill_without,
            "yearly_bill_with_spp": yearly_bill_with,
            "yearly_savings": _round2(yearly_bill_without - yearly_bill_with),
        }

    output = {"result": result_payload, "variants": variants}
    if battery:
        output["battery"] = battery
        output["outage_support"] = {
            "critical_load_supported_kw": battery["critical_load_kw"],
            "backup_hours": battery["backup_hours"],
        }
    return output


def _build_variants(base: BaseSystemResult, yield_factor: float, tariff: float) -> list[dict[str, Any]]:
    annual_need = 0.0
    if base.coverage_percent > 0:
        annual_need = (base.annual_generation * 100) / base.coverage_percent

    def make_variant(name: str, factor: float, with_battery: bool = False) -> dict[str, Any]:
        res = _build_base_result(
            system_kw=base.system_kw * factor,
            annual_kwh_need=annual_need,
            tariff=tariff,
            yield_factor=yield_factor,
        )
        out = {
            "name": name,
            **res.__dict__,
            "need_battery": with_battery,
        }
        if with_battery:
            out["battery_kwh"] = _round2(max(res.system_kw * 0.3 * 4, 5.0))
        return out

    return [
        make_variant("economy", 0.5),
        make_variant("optimal", 0.75),
        make_variant("premium", 1.0, with_battery=True),
    ]
