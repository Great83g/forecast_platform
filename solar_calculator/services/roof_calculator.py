from .calculator_engine import PANEL_AREA_WITH_GAPS_M2, PANEL_POWER_W, ROOF_COEFFICIENTS, SHADING_COEFFICIENTS


def calculate_roof_capacity_kw(*, roof_area: float, roof_type: str, shading: str) -> float:
    roof_coef = ROOF_COEFFICIENTS.get(roof_type, ROOF_COEFFICIENTS["simple"])
    shading_coef = SHADING_COEFFICIENTS.get(shading, SHADING_COEFFICIENTS["none"])
    usable_area = max(roof_area, 0) * roof_coef * shading_coef
    panel_count = int(usable_area // PANEL_AREA_WITH_GAPS_M2)
    return panel_count * PANEL_POWER_W / 1000
