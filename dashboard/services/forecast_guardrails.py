from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

INSTALLED_MW_1_2 = 1.2
MAX_GUARDRAIL_MW_1_2 = INSTALLED_MW_1_2 * 1.05
GUARDRAIL_REASON_OK = "OK"
GUARDRAIL_REASON_BAD_IRRADIATION = "FALLBACK_BAD_IRRADIATION"

SUMMER_PROFILE = {
    6: 0.10,
    7: 0.25,
    8: 0.45,
    9: 0.65,
    10: 0.80,
    11: 0.90,
    12: 0.95,
    13: 0.95,
    14: 0.85,
    15: 0.70,
    16: 0.50,
    17: 0.30,
    18: 0.15,
    19: 0.05,
    20: 0.00,
}

SHOULDER_PROFILE = {
    6: 0.02,
    7: 0.10,
    8: 0.25,
    9: 0.45,
    10: 0.60,
    11: 0.75,
    12: 0.80,
    13: 0.80,
    14: 0.65,
    15: 0.45,
    16: 0.25,
    17: 0.10,
    18: 0.03,
    19: 0.00,
    20: 0.00,
}

WINTER_PROFILE = {
    6: 0.00,
    7: 0.00,
    8: 0.05,
    9: 0.15,
    10: 0.30,
    11: 0.45,
    12: 0.55,
    13: 0.55,
    14: 0.40,
    15: 0.25,
    16: 0.10,
    17: 0.02,
    18: 0.00,
    19: 0.00,
    20: 0.00,
}


def _timestamp(value: Any) -> pd.Timestamp:
    return pd.to_datetime(value, errors="coerce")


def is_bad_irradiation(value: Any, hour: int) -> bool:
    if hour < 6 or hour > 20:
        return False
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return True
    return float(numeric) <= 0 or float(numeric) > 1300


def fallback_heuristic_1_2mw(timestamp: Any) -> float:
    ts = _timestamp(timestamp)
    if pd.isna(ts):
        return 0.0
    hour = int(ts.hour)
    if hour < 6 or hour > 20:
        return 0.0
    month = int(ts.month)
    if 5 <= month <= 9:
        profile = SUMMER_PROFILE
    elif month in {3, 4, 10}:
        profile = SHOULDER_PROFILE
    else:
        profile = WINTER_PROFILE
    return round(INSTALLED_MW_1_2 * profile.get(hour, 0.0), 3)


def apply_visual_crossing_fallback(df: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in df.columns:
        raise ValueError("timestamp column is required for Visual Crossing fallback")
    if "pred_final_mw" not in df.columns:
        raise ValueError("pred_final_mw column is required for Visual Crossing fallback")

    guarded = df.copy()
    if "irradiation" not in guarded.columns:
        guarded["irradiation"] = pd.NA

    guarded["pred_final_raw_mw"] = pd.to_numeric(guarded["pred_final_mw"], errors="coerce")
    guarded["guardrail_reason"] = GUARDRAIL_REASON_OK

    for idx, row in guarded.iterrows():
        ts = _timestamp(row["timestamp"])
        hour = 0 if pd.isna(ts) else int(ts.hour)
        raw_value = guarded.at[idx, "pred_final_raw_mw"]
        if is_bad_irradiation(row.get("irradiation"), hour):
            guarded.at[idx, "pred_final_mw"] = fallback_heuristic_1_2mw(ts)
            guarded.at[idx, "guardrail_reason"] = GUARDRAIL_REASON_BAD_IRRADIATION
        else:
            guarded.at[idx, "pred_final_mw"] = raw_value

    guarded["pred_final_mw"] = (
        pd.to_numeric(guarded["pred_final_mw"], errors="coerce")
        .fillna(0.0)
        .clip(lower=0.0, upper=MAX_GUARDRAIL_MW_1_2)
    )
    return guarded


def write_forecast_guardrail_log(df: pd.DataFrame, path: str | Path = "forecast_guardrail_log.csv") -> int:
    columns = ["timestamp", "irradiation", "pred_final_raw_mw", "pred_final_mw", "guardrail_reason"]
    if "guardrail_reason" not in df.columns:
        pd.DataFrame(columns=columns).to_csv(path, index=False)
        return 0
    log_df = df.loc[df["guardrail_reason"] != GUARDRAIL_REASON_OK].copy()
    for column in columns:
        if column not in log_df.columns:
            log_df[column] = pd.NA
    log_df.to_csv(path, index=False, columns=columns)
    return len(log_df)
