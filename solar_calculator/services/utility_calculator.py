from __future__ import annotations

from .calculator_engine import PANEL_POWER_W


def calculate_by_target_power(target_mw: float, specific_yield: float) -> dict:
    target_mw = max(target_mw, 0.0)
    dc_ratio = 1.2
    dc_mw = target_mw * dc_ratio
    panels = int(round((dc_mw * 1_000_000) / PANEL_POWER_W)) if dc_mw else 0
    annual_generation_gwh = target_mw * specific_yield / 1000
    inverter_count = max(int(round(target_mw * 4)), 1) if target_mw else 0
    return {
        "result": {
            "AC_MW": round(target_mw, 3),
            "DC_MW": round(dc_mw, 3),
            "panels": panels,
            "land_required": round(target_mw * 1.5, 3),
            "annual_generation_GWh": round(annual_generation_gwh, 3),
            "inverter_count": inverter_count,
        },
        "variants": [],
    }


def calculate_by_land(land_hectares: float, specific_yield: float) -> dict:
    land_hectares = max(land_hectares, 0.0)
    mw_possible = land_hectares / 1.5
    return calculate_by_target_power(mw_possible, specific_yield)
