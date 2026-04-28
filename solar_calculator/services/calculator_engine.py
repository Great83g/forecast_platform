from __future__ import annotations

import math
from typing import Any

PANEL_MODEL = "SP-N16/144HG"
PANEL_POWER_W = 580
PANEL_LENGTH_M = 2.278
PANEL_WIDTH_M = 1.134
PANEL_AREA_M2 = 2.58
PANEL_AREA_WITH_GAP_M2 = 3.10
PANEL_WEIGHT_KG = 32

DEFAULT_SPECIFIC_YIELD = 1450.0
DEFAULT_TARIFF_KZT_PER_KWH = 35.0
PANEL_PRICE_PER_W_KZT = 100.0
DEFAULT_COST_PER_KW = PANEL_PRICE_PER_W_KZT * 1000

PANEL_PRICE_KZT = 58_000.0
INVERTER_COST_PER_KW_KZT = 70_000.0
MOUNTING_COST_PER_KW_KZT = 50_000.0
CABLES_PROTECTION_COST_PER_KW_KZT = 30_000.0
BATTERY_COST_PER_KWH_KZT = 120_000.0

DC_AC_RATIO = 1.2
LAND_PER_MW_HA = 1.5
INVERTER_UNIT_MW = 0.25

CALC_MODES = [
    "consumption",
    "appliances",
    "budget",
    "roof_area",
    "max_roof",
    "utility_power",
    "utility_land",
]

RESIDENTIAL_VARIANT_MODES = {"consumption", "appliances", "budget"}

ROOF_COEFFICIENTS = {
    "flat": 0.80,
    "simple": 0.70,
    "complex": 0.50,
}


def _round2(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 2)


def _to_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _meta() -> dict[str, Any]:
    return {
        "panel_model": PANEL_MODEL,
        "panel_power_w": PANEL_POWER_W,
        "panel_length_m": PANEL_LENGTH_M,
        "panel_width_m": PANEL_WIDTH_M,
        "area_panel_m2": PANEL_AREA_M2,
        "area_with_gap_m2": PANEL_AREA_WITH_GAP_M2,
        "weight_kg": PANEL_WEIGHT_KG,
    }


def _response(mode: str, *, result_type: str = "residential", inputs_echo: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "mode": mode,
        "result_type": result_type,
        "meta": _meta(),
        "inputs_echo": inputs_echo or {},
        "result": {},
        "variants": [],
        "warnings": [],
        "errors": [],
    }


def _roof_fit(roof_area_m2: float | None, area_required: float) -> tuple[str | None, str | None]:
    if roof_area_m2 is None:
        return None, None

    if roof_area_m2 >= area_required:
        return "fits", f"Система требует {_round2(area_required)} м², указанной площади достаточно."
    if roof_area_m2 >= area_required * 0.85:
        return "partial", f"Система требует {_round2(area_required)} м², указанной площади немного не хватает."
    return "not_fit", f"Система требует {_round2(area_required)} м², указанной площади недостаточно."


def _residential_from_panel_count(
    *,
    panel_count: int,
    specific_yield: float,
    tariff: float,
    cost_per_kw: float,
    annual_kwh_need: float | None = None,
    roof_area_m2: float | None = None,
    basis: str,
    summary: str,
    self_consumption_percent: float = 100.0,
    battery_kwh: float = 0.0,
) -> dict[str, Any]:
    system_kw_final = panel_count * PANEL_POWER_W / 1000
    area_panels = panel_count * PANEL_AREA_M2
    area_required = panel_count * PANEL_AREA_WITH_GAP_M2
    weight = panel_count * PANEL_WEIGHT_KG
    annual_generation = system_kw_final * specific_yield
    monthly_generation = annual_generation / 12

    usable_generation_kwh = annual_generation * max(min(self_consumption_percent, 100.0), 0.0) / 100
    export_or_unused_kwh = max(annual_generation - usable_generation_kwh, 0.0)

    coverage_percent = None
    yearly_bill_without_spp = None
    yearly_bill_with_spp = None
    yearly_savings = None
    payback_years = None

    if annual_kwh_need and annual_kwh_need > 0:
        coverage_percent = min(100.0, usable_generation_kwh / annual_kwh_need * 100)
        yearly_bill_without_spp = annual_kwh_need * tariff
        yearly_bill_with_spp = max(0.0, (annual_kwh_need - usable_generation_kwh) * tariff)
        yearly_savings = yearly_bill_without_spp - yearly_bill_with_spp

    panels_cost_kzt = panel_count * PANEL_PRICE_KZT
    inverter_cost_kzt = system_kw_final * INVERTER_COST_PER_KW_KZT
    mounting_cost_kzt = system_kw_final * MOUNTING_COST_PER_KW_KZT
    cables_protection_cost_kzt = system_kw_final * CABLES_PROTECTION_COST_PER_KW_KZT
    battery_cost_kzt = max(battery_kwh, 0.0) * BATTERY_COST_PER_KWH_KZT
    estimated_cost = panels_cost_kzt + inverter_cost_kzt + mounting_cost_kzt + cables_protection_cost_kzt + battery_cost_kzt

    if yearly_savings and yearly_savings > 0:
        payback_years = estimated_cost / yearly_savings

    roof_fit, roof_fit_message = _roof_fit(roof_area_m2, area_required)

    return {
        "system_kw": _round2(system_kw_final),
        "panel_count": int(panel_count),
        "area_panels_m2": _round2(area_panels),
        "area_required_m2": _round2(area_required),
        "weight_kg": _round2(weight),
        "inverter_size_kw": _round2(system_kw_final * 0.8),
        "annual_generation_kwh": _round2(annual_generation),
        "monthly_generation_kwh": _round2(monthly_generation),
        "coverage_percent": _round2(coverage_percent),
        "estimated_cost_kzt": _round2(estimated_cost),
        "yearly_bill_without_spp_kzt": _round2(yearly_bill_without_spp),
        "yearly_bill_with_spp_kzt": _round2(yearly_bill_with_spp),
        "yearly_savings_kzt": _round2(yearly_savings),
        "payback_years": _round2(payback_years),
        "roof_fit": roof_fit,
        "roof_fit_message": roof_fit_message,
        "summary": summary,
        "calculation_basis": basis,
        "cost_breakdown": {
            "panels_cost_kzt": _round2(panels_cost_kzt),
            "inverter_cost_kzt": _round2(inverter_cost_kzt),
            "mounting_cost_kzt": _round2(mounting_cost_kzt),
            "cables_protection_cost_kzt": _round2(cables_protection_cost_kzt),
            "battery_cost_kzt": _round2(battery_cost_kzt),
            "total_cost_kzt": _round2(estimated_cost),
        },
        "energy_model": {
            "self_consumption_percent": _round2(self_consumption_percent),
            "usable_generation_kwh": _round2(usable_generation_kwh),
            "export_or_unused_kwh": _round2(export_or_unused_kwh),
            "annual_kwh_need": _round2(annual_kwh_need),
        },
        "battery_kwh": _round2(battery_kwh) if battery_kwh else 0.0,
    }


def _build_variants(*, target_system_kw: float, annual_kwh_need: float | None, specific_yield: float, tariff: float, cost_per_kw: float) -> list[dict[str, Any]]:
    variants = []
    monthly_kwh = (annual_kwh_need / 12) if annual_kwh_need else 0.0

    for name, factor, self_cons in (("economy", 0.5, 60.0), ("optimal", 0.75, 75.0), ("premium", 1.0, 90.0)):
        target_panels = max(1, int(round((target_system_kw * factor) * 1000 / PANEL_POWER_W)))
        battery_kwh = 0.0
        if name == "premium":
            daily_kwh = (monthly_kwh / 30) if monthly_kwh > 0 else ((target_panels * PANEL_POWER_W / 1000) * specific_yield / 12 / 30)
            battery_kwh = max(5.0, float(math.ceil(daily_kwh * 0.4)))

        item = _residential_from_panel_count(
            panel_count=target_panels,
            specific_yield=specific_yield,
            tariff=tariff,
            cost_per_kw=cost_per_kw,
            annual_kwh_need=annual_kwh_need,
            roof_area_m2=None,
            basis=f"{name} ({int(factor * 100)}% от целевой мощности)",
            summary=f"Вариант {name}: {target_panels} панелей, {_round2(target_panels * PANEL_POWER_W / 1000)} кВт.",
            self_consumption_percent=self_cons,
            battery_kwh=battery_kwh,
        )
        if name == "premium":
            item["need_battery"] = True
            item["note"] = "Premium включает аккумулятор %s кВт·ч. Он повышает долю собственного потребления до 90%, но увеличивает стоимость системы." % int(battery_kwh)
        else:
            item["need_battery"] = False
            item["note"] = "Без аккумулятора часть дневной генерации может уходить в сеть или не использоваться."
        item["name"] = name
        variants.append(item)
    return variants


def _validate_positive(value: float | None, field: str, errors: list[str], *, allow_zero: bool = False) -> None:
    if value is None:
        errors.append(f"Поле '{field}' обязательно.")
        return
    if allow_zero:
        if value < 0:
            errors.append(f"Поле '{field}' должно быть >= 0.")
    elif value <= 0:
        errors.append(f"Поле '{field}' должно быть > 0.")


def _label_project_size(ac_mw: float) -> str:
    if ac_mw < 1:
        return "small"
    if ac_mw <= 10:
        return "medium"
    if ac_mw <= 50:
        return "large"
    return "utility_scale"


def _station_price_range(ac_mw: float) -> tuple[float, float]:
    _ = ac_mw
    # Базируемся на текущей цене панели: 100 тг/Вт (58 000 тг за 580 Вт).
    station_price_per_kw = PANEL_PRICE_PER_W_KZT * 1000
    return station_price_per_kw, station_price_per_kw


def _build_station_economics(*, response: dict[str, Any], ac_mw: float, annual_generation_kwh: float, tariff_value: float | None) -> dict[str, Any]:
    price_min_kw, price_max_kw = _station_price_range(ac_mw)
    ac_kw = ac_mw * 1000
    estimated_cost_min = ac_kw * price_min_kw
    estimated_cost_max = ac_kw * price_max_kw
    estimated_cost_mid = (estimated_cost_min + estimated_cost_max) / 2

    result = {
        "price_per_kw_kzt": {
            "min": _round2(price_min_kw),
            "max": _round2(price_max_kw),
            "mid": _round2((price_min_kw + price_max_kw) / 2),
        },
        "estimated_cost_min_kzt": _round2(estimated_cost_min),
        "estimated_cost_max_kzt": _round2(estimated_cost_max),
    }

    if tariff_value is not None and tariff_value > 0:
        yearly_savings = annual_generation_kwh * tariff_value
        payback_years = estimated_cost_mid / yearly_savings if yearly_savings > 0 else None
        result["yearly_savings_kzt"] = _round2(yearly_savings)
        result["payback_years"] = _round2(payback_years)
    else:
        response["warnings"].append("Экономика не рассчитана: укажите tariff_kzt_per_kwh для station mode.")
        result["yearly_savings_kzt"] = None
        result["payback_years"] = None

    return result


def calculate(mode: str, inputs: dict[str, Any]) -> dict[str, Any]:
    if mode not in CALC_MODES:
        return {
            "mode": mode,
            "result_type": "unknown",
            "meta": _meta(),
            "inputs_echo": inputs,
            "result": {},
            "variants": [],
            "warnings": [],
            "errors": [f"Unsupported mode: {mode}"],
        }

    response = _response(mode, result_type="utility" if mode.startswith("utility") else "residential", inputs_echo=inputs)

    specific_yield = _to_float(inputs.get("specific_yield"), DEFAULT_SPECIFIC_YIELD) or DEFAULT_SPECIFIC_YIELD
    tariff = _to_float(inputs.get("tariff_kzt_per_kwh"), DEFAULT_TARIFF_KZT_PER_KWH) or DEFAULT_TARIFF_KZT_PER_KWH
    cost_per_kw = _to_float(inputs.get("cost_per_kw"), DEFAULT_COST_PER_KW) or DEFAULT_COST_PER_KW

    if mode == "consumption":
        monthly_kwh = _to_float(inputs.get("monthly_kwh"))
        roof_area_m2 = _to_float(inputs.get("roof_area_m2"))
        _validate_positive(monthly_kwh, "monthly_kwh", response["errors"])
        if roof_area_m2 is not None and roof_area_m2 < 3:
            response["warnings"].append("Площадь крыши очень маленькая для полноценной системы.")
        if response["errors"]:
            return response

        annual_kwh = monthly_kwh * 12
        system_kw_target = annual_kwh / specific_yield
        panel_count = int(math.ceil(system_kw_target * 1000 / PANEL_POWER_W))
        response["result"] = _residential_from_panel_count(
            panel_count=panel_count,
            specific_yield=specific_yield,
            tariff=tariff,
            cost_per_kw=cost_per_kw,
            annual_kwh_need=annual_kwh,
            roof_area_m2=roof_area_m2,
            basis="По среднемесячному потреблению и удельной генерации.",
            summary=f"Для потребления {_round2(monthly_kwh)} кВт·ч/мес подходит система на {_round2(panel_count * PANEL_POWER_W / 1000)} кВт из {panel_count} панелей.",
        )
        response["result"]["annual_kwh_need"] = _round2(annual_kwh)
        response["warnings"].append("Расчёт ориентировочный. Без почасового профиля потребления фактическая экономия может отличаться.")
        response["variants"] = _build_variants(
            target_system_kw=panel_count * PANEL_POWER_W / 1000,
            annual_kwh_need=annual_kwh,
            specific_yield=specific_yield,
            tariff=tariff,
            cost_per_kw=cost_per_kw,
        )
        return response

    if mode == "roof_area":
        roof_area_m2 = _to_float(inputs.get("roof_area_m2"))
        roof_type = str(inputs.get("roof_type") or "simple")
        monthly_kwh = _to_float(inputs.get("monthly_kwh"))
        annual_kwh_need = monthly_kwh * 12 if monthly_kwh and monthly_kwh > 0 else None

        _validate_positive(roof_area_m2, "roof_area_m2", response["errors"])
        if roof_area_m2 is not None and roof_area_m2 < 3:
            response["errors"].append("Для режима roof_area минимальная площадь крыши 3 м².")
        if roof_type not in ROOF_COEFFICIENTS:
            response["warnings"].append("roof_type неизвестен, применён коэффициент simple.")
            roof_type = "simple"
        if response["errors"]:
            return response

        usable_area = roof_area_m2 * ROOF_COEFFICIENTS[roof_type]
        panel_count = int(math.floor(usable_area / PANEL_AREA_WITH_GAP_M2))
        panel_count = max(panel_count, 0)
        summary = f"На крыше {_round2(roof_area_m2)} м² можно разместить систему примерно на {_round2(panel_count * PANEL_POWER_W / 1000)} кВт."
        if annual_kwh_need:
            summary += f" При потреблении {_round2(monthly_kwh)} кВт·ч/мес рассчитаны покрытие и экономика."

        result = _residential_from_panel_count(
            panel_count=panel_count,
            specific_yield=specific_yield,
            tariff=tariff,
            cost_per_kw=cost_per_kw,
            annual_kwh_need=annual_kwh_need,
            roof_area_m2=roof_area_m2,
            basis="По площади крыши и коэффициенту типа кровли.",
            summary=summary,
        )
        result["usable_area_m2"] = _round2(usable_area)
        result["free_area_m2"] = _round2(max(0.0, roof_area_m2 - (panel_count * PANEL_AREA_WITH_GAP_M2)))
        result["roof_fit"] = "fits"
        result["roof_fit_message"] = "Расчёт построен от доступной площади крыши."
        if annual_kwh_need:
            result["annual_kwh_need"] = _round2(annual_kwh_need)
            if (result.get("annual_generation_kwh") or 0) > annual_kwh_need:
                response["warnings"].append("Генерация станции превышает годовое потребление объекта.")
        response["result"] = result
        if annual_kwh_need:
            response["warnings"].append("Расчёт ориентировочный. Без почасового профиля потребления фактическая экономия может отличаться.")
        return response

    if mode == "max_roof":
        roof_area_m2 = _to_float(inputs.get("roof_area_m2"))
        roof_type = str(inputs.get("roof_type") or "simple")
        _validate_positive(roof_area_m2, "roof_area_m2", response["errors"])
        if roof_area_m2 is not None and roof_area_m2 < 3:
            response["errors"].append("Для режима max_roof минимальная площадь крыши 3 м².")
        if roof_type not in ROOF_COEFFICIENTS:
            roof_type = "simple"
        if response["errors"]:
            return response

        usable_area = roof_area_m2 * ROOF_COEFFICIENTS[roof_type]
        panel_count = int(math.floor(usable_area / PANEL_AREA_WITH_GAP_M2))
        system_kw = panel_count * PANEL_POWER_W / 1000
        annual_generation = system_kw * specific_yield
        response["result"] = {
            "max_panel_count": panel_count,
            "max_system_kw": _round2(system_kw),
            "annual_generation_kwh": _round2(annual_generation),
            "occupied_area_m2": _round2(panel_count * PANEL_AREA_WITH_GAP_M2),
            "free_area_m2": _round2(max(0.0, roof_area_m2 - panel_count * PANEL_AREA_WITH_GAP_M2)),
            "roof_fit": "fits",
            "roof_fit_message": "Максимально возможная конфигурация под указанную крышу.",
            "summary": f"Максимум для крыши {_round2(roof_area_m2)} м²: {panel_count} панелей и {_round2(system_kw)} кВт.",
            "calculation_basis": "Максимизация количества панелей по полезной площади крыши.",
        }
        return response

    if mode == "budget":
        budget_kzt = _to_float(inputs.get("budget_kzt"))
        _validate_positive(budget_kzt, "budget_kzt", response["errors"])
        if response["errors"]:
            return response

        system_kw_raw = budget_kzt / cost_per_kw
        panel_count = int(math.floor(system_kw_raw * 1000 / PANEL_POWER_W))
        if panel_count < 1:
            response["errors"].append("Бюджет слишком мал: не хватает даже на 1 панель.")
            response["warnings"].append("Увеличьте бюджет или уменьшите стоимость за кВт.")
            return response

        result = _residential_from_panel_count(
            panel_count=panel_count,
            specific_yield=specific_yield,
            tariff=tariff,
            cost_per_kw=cost_per_kw,
            annual_kwh_need=None,
            roof_area_m2=None,
            basis="По бюджету и стоимости кВт установленной мощности.",
            summary=f"При бюджете {_round2(budget_kzt)} тг доступна система {_round2(panel_count * PANEL_POWER_W / 1000)} кВт ({panel_count} панелей).",
        )
        result["system_kw_raw"] = _round2(system_kw_raw)
        response["result"] = result
        response["warnings"].append("Расчёт ориентировочный. Без почасового профиля потребления фактическая экономия может отличаться.")
        response["variants"] = _build_variants(
            target_system_kw=panel_count * PANEL_POWER_W / 1000,
            annual_kwh_need=None,
            specific_yield=specific_yield,
            tariff=tariff,
            cost_per_kw=cost_per_kw,
        )
        return response

    if mode == "utility_power":
        target_mw_ac = _to_float(inputs.get("target_mw_ac"))
        tariff_station = _to_float(inputs.get("tariff_kzt_per_kwh"))
        _validate_positive(target_mw_ac, "target_mw_ac", response["errors"])
        if response["errors"]:
            return response

        ac_mw = target_mw_ac
        dc_mw = ac_mw * DC_AC_RATIO
        panels = int(math.ceil(dc_mw * 1_000_000 / PANEL_POWER_W))
        annual_generation_gwh = ac_mw * specific_yield / 1000
        annual_generation_kwh = annual_generation_gwh * 1_000_000
        land_required_ha = ac_mw * LAND_PER_MW_HA
        economics = _build_station_economics(
            response=response,
            ac_mw=ac_mw,
            annual_generation_kwh=annual_generation_kwh,
            tariff_value=tariff_station,
        )

        summary = (
            f"Станция {_round2(ac_mw)} МВт AC: участок ~{_round2(land_required_ha)} га, "
            f"генерация ~{_round2(annual_generation_gwh)} ГВт·ч/год, "
            f"стоимость ~{economics['estimated_cost_min_kzt']}-{economics['estimated_cost_max_kzt']} тг."
        )

        result = {
            "ac_mw": _round2(ac_mw),
            "dc_mw": _round2(dc_mw),
            "panels": panels,
            "land_required_ha": _round2(land_required_ha),
            "annual_generation_gwh": _round2(annual_generation_gwh),
            "annual_generation_kwh": _round2(annual_generation_kwh),
            "inverter_count": int(math.ceil(ac_mw / INVERTER_UNIT_MW)),
            "summary": summary,
            "calculation_basis": "Расчёт station-проекта по целевой AC мощности.",
            **economics,
        }
        if ac_mw < 1:
            result["ac_kw"] = _round2(ac_mw * 1000)
            result["dc_kw"] = _round2(dc_mw * 1000)
        response["result"] = result
        return response

    if mode == "utility_land":
        land_hectares = _to_float(inputs.get("land_hectares"))
        tariff_station = _to_float(inputs.get("tariff_kzt_per_kwh"))
        _validate_positive(land_hectares, "land_hectares", response["errors"])
        if land_hectares and land_hectares > 100:
            response["warnings"].append("Рассчитывается очень крупная промышленная СЭС.")
        if response["errors"]:
            return response

        ac_mw = land_hectares / LAND_PER_MW_HA
        dc_mw = ac_mw * DC_AC_RATIO
        panels = int(math.ceil(dc_mw * 1_000_000 / PANEL_POWER_W))
        annual_generation_gwh = ac_mw * specific_yield / 1000
        annual_generation_kwh = annual_generation_gwh * 1_000_000
        economics = _build_station_economics(
            response=response,
            ac_mw=ac_mw,
            annual_generation_kwh=annual_generation_kwh,
            tariff_value=tariff_station,
        )

        result = {
            "ac_mw": _round2(ac_mw),
            "dc_mw": _round2(dc_mw),
            "panels": panels,
            "land_hectares": _round2(land_hectares),
            "annual_generation_gwh": _round2(annual_generation_gwh),
            "annual_generation_kwh": _round2(annual_generation_kwh),
            "inverter_count": int(math.ceil(ac_mw / INVERTER_UNIT_MW)),
            "project_size_label": _label_project_size(ac_mw),
            "summary": (
                f"На участке {_round2(land_hectares)} га можно построить станцию примерно на {_round2(ac_mw)} МВт, "
                f"с генерацией ~{_round2(annual_generation_gwh)} ГВт·ч/год и стоимостью "
                f"~{economics['estimated_cost_min_kzt']}-{economics['estimated_cost_max_kzt']} тг."
            ),
            "calculation_basis": "Расчёт station-проекта по площади земельного участка.",
            **economics,
        }
        if ac_mw < 1:
            result["ac_kw"] = _round2(ac_mw * 1000)
            result["dc_kw"] = _round2(dc_mw * 1000)
        response["result"] = result
        return response

    # appliances
    appliances = inputs.get("appliances")
    if not isinstance(appliances, list) or not appliances:
        response["errors"].append("Поле 'appliances' должно быть непустым массивом.")
        return response

    daily_kwh = 0.0
    peak_kw = 0.0
    for i, item in enumerate(appliances):
        power_kw = _to_float(item.get("power_kw"))
        hours_per_day = _to_float(item.get("hours_per_day"))
        quantity = _to_float(item.get("quantity"))
        _validate_positive(power_kw, f"appliances[{i}].power_kw", response["errors"])
        _validate_positive(hours_per_day, f"appliances[{i}].hours_per_day", response["errors"], allow_zero=True)
        _validate_positive(quantity, f"appliances[{i}].quantity", response["errors"])
        if quantity is not None and quantity < 1:
            response["errors"].append(f"appliances[{i}].quantity должно быть >= 1.")
        if power_kw and hours_per_day is not None and quantity:
            daily_kwh += power_kw * hours_per_day * quantity
            peak_kw += power_kw * quantity

    if response["errors"]:
        return response

    monthly_kwh = daily_kwh * 30
    annual_kwh = monthly_kwh * 12
    target_system_kw = annual_kwh / specific_yield
    panel_count = int(math.ceil(target_system_kw * 1000 / PANEL_POWER_W))

    result = _residential_from_panel_count(
        panel_count=panel_count,
        specific_yield=specific_yield,
        tariff=tariff,
        cost_per_kw=cost_per_kw,
        annual_kwh_need=annual_kwh,
        roof_area_m2=_to_float(inputs.get("roof_area_m2")),
        basis="По списку приборов, часам работы и количеству.",
        summary=f"По приборной нагрузке подходит система на {_round2(panel_count * PANEL_POWER_W / 1000)} кВт из {panel_count} панелей.",
    )
    result["daily_kwh"] = _round2(daily_kwh)
    result["monthly_kwh_derived"] = _round2(monthly_kwh)
    result["annual_kwh_derived"] = _round2(annual_kwh)
    result["peak_kw"] = _round2(peak_kw)
    response["result"] = result
    response["warnings"].append("Расчёт ориентировочный. Без почасового профиля потребления фактическая экономия может отличаться.")
    response["variants"] = _build_variants(
        target_system_kw=panel_count * PANEL_POWER_W / 1000,
        annual_kwh_need=annual_kwh,
        specific_yield=specific_yield,
        tariff=tariff,
        cost_per_kw=cost_per_kw,
    )
    return response
