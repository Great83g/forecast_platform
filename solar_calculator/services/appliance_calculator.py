from __future__ import annotations

from typing import Any


def calculate_from_appliances(appliances: list[dict[str, Any]]) -> dict[str, float]:
    daily_kwh = 0.0
    peak_kw = 0.0
    for item in appliances:
        power_kw = float(item.get("power_kw") or 0)
        hours_per_day = float(item.get("hours_per_day") or 0)
        qty = float(item.get("quantity") or 1)
        simultaneity = float(item.get("simultaneity_factor") or 0.7)
        daily_kwh += power_kw * hours_per_day * qty
        peak_kw += power_kw * qty * simultaneity

    return {
        "daily_kwh": round(daily_kwh, 2),
        "monthly_kwh": round(daily_kwh * 30, 2),
        "peak_kw": round(peak_kw, 2),
    }
