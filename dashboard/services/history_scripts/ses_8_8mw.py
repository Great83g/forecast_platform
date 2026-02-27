from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

TIME_RE = re.compile(r"^\s*\d{1,2}:\d{2}\s*$")
EXCLUDE_TIME_RE = re.compile(r"(?:прогноз|scada|аскуэ)", re.IGNORECASE)

COL_TIME = 0
COL_POWER_MW = 2
COL_IRR = 3
COL_AIR_TEMP = 6
COL_PV_TEMP = 7

MIN_POWER_KW = 0.0001
POWER_UPPER_BAD_MW = 9.0
IRR_MAX = 1200.0
PV_TEMP_MIN = -40.0
PV_TEMP_MAX = 110.0


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["ds", "irradiation", "air_temp", "pv_temp", "power_kw"])


def _parse_sheet_date(sheet_name: str) -> pd.Timestamp | None:
    cleaned = sheet_name.strip().replace("г", "").replace(" ", "")
    try:
        return pd.to_datetime(cleaned, format="%d.%m.%Y", errors="raise")
    except Exception:
        return None


def _read_sheet_rows(file_path: Path, sheet_name: str, day_ts: pd.Timestamp) -> pd.DataFrame:
    raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    if raw is None or raw.empty or COL_TIME not in raw.columns:
        return _empty_df()

    col_time = raw[COL_TIME].astype(str).str.strip()
    mask_time = col_time.str.match(TIME_RE) & ~col_time.str.contains(EXCLUDE_TIME_RE, na=False)
    block = raw.loc[mask_time].copy()
    if block.empty:
        return _empty_df()

    ds = pd.to_datetime(day_ts.strftime("%Y-%m-%d") + " " + block[COL_TIME].astype(str).str.strip(), errors="coerce")

    out = pd.DataFrame(
        {
            "ds": ds,
            "power_mw": pd.to_numeric(block.get(COL_POWER_MW), errors="coerce"),
            "irradiation": pd.to_numeric(block.get(COL_IRR), errors="coerce"),
            "air_temp": pd.to_numeric(block.get(COL_AIR_TEMP), errors="coerce"),
            "pv_temp": pd.to_numeric(block.get(COL_PV_TEMP), errors="coerce"),
        }
    )
    return out.dropna(subset=["ds"])


def build_history_dataframe(station) -> pd.DataFrame:
    folder = Path(getattr(station, "auto_history_folder", "") or "")
    if not folder.exists():
        return _empty_df()

    excel_files = sorted([p for p in folder.glob("*.xlsx") if not p.name.startswith("~$")])
    if not excel_files:
        return _empty_df()

    parts: list[pd.DataFrame] = []
    for file_path in excel_files:
        try:
            xls = pd.ExcelFile(file_path)
        except Exception:
            continue

        for sheet_name in xls.sheet_names:
            day_ts = _parse_sheet_date(sheet_name)
            if day_ts is None:
                continue
            part = _read_sheet_rows(file_path, sheet_name, day_ts)
            if not part.empty:
                parts.append(part)

    if not parts:
        return _empty_df()

    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values("ds").drop_duplicates(subset=["ds"], keep="last").reset_index(drop=True)

    df.loc[(df["power_mw"] < 0) | (df["power_mw"] > POWER_UPPER_BAD_MW), "power_mw"] = pd.NA
    df.loc[(df["irradiation"] < 0) | (df["irradiation"] > IRR_MAX), "irradiation"] = pd.NA
    df.loc[(df["pv_temp"] < PV_TEMP_MIN) | (df["pv_temp"] > PV_TEMP_MAX), "pv_temp"] = pd.NA

    df["power_kw"] = df["power_mw"] * 1000.0

    hourly = (
        df.set_index("ds")[["irradiation", "air_temp", "pv_temp", "power_kw"]]
        .resample("h")
        .mean()
        .reset_index()
    )
    hourly = hourly[hourly["power_kw"].fillna(0) > MIN_POWER_KW].copy()
    return hourly[["ds", "irradiation", "air_temp", "pv_temp", "power_kw"]].reset_index(drop=True)
