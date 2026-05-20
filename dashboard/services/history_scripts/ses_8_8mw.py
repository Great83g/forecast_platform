from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

TIME_RE = re.compile(r"^\s*\d{1,2}:\d{2}(?::\d{2})?\s*$")
EXCLUDE_TIME_RE = re.compile(r"(?:прогноз|scada|аскуэ)", re.IGNORECASE)

COL_TIME = 0
COL_ENERGY_KWH = 1
COL_POWER_MW = 2
COL_IRR = 3
COL_AIR_TEMP = 6
COL_PV_TEMP = 7

MIN_POWER_KW = 0.0001
HISTORY_SHIFT_HOURS = 1
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




def _normalize_time_token(value) -> str | None:
    if pd.isna(value):
        return None

    raw = str(value).strip()
    if not raw:
        return None
    if EXCLUDE_TIME_RE.search(raw):
        return None

    if TIME_RE.match(raw):
        parts = raw.split(":")
        return f"{int(parts[0]):02d}:{int(parts[1]):02d}"

    ts = pd.to_datetime(value, errors="coerce")
    if pd.notna(ts):
        return ts.strftime("%H:%M")

    if isinstance(value, (int, float)) and 0 <= float(value) < 1:
        total_minutes = int(round(float(value) * 24 * 60))
        hh = (total_minutes // 60) % 24
        mm = total_minutes % 60
        return f"{hh:02d}:{mm:02d}"

    return None

def _read_sheet_rows(file_path: Path, sheet_name: str, day_ts: pd.Timestamp) -> pd.DataFrame:
    raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
    if raw is None or raw.empty or COL_TIME not in raw.columns:
        return _empty_df()

    normalized_time = raw[COL_TIME].apply(_normalize_time_token)
    mask_time = normalized_time.notna()
    block = raw.loc[mask_time].copy()
    if block.empty:
        return _empty_df()

    block["_time_norm"] = normalized_time.loc[mask_time].astype(str)
    ds = pd.to_datetime(day_ts.strftime("%Y-%m-%d") + " " + block["_time_norm"], errors="coerce")

    out = pd.DataFrame(
        {
            "ds": ds,
            "energy_kwh": pd.to_numeric(block.get(COL_ENERGY_KWH), errors="coerce"),
            "power_mw": pd.to_numeric(block.get(COL_POWER_MW), errors="coerce"),
            "irradiation": pd.to_numeric(block.get(COL_IRR), errors="coerce"),
            "air_temp": pd.to_numeric(block.get(COL_AIR_TEMP), errors="coerce"),
            "pv_temp": pd.to_numeric(block.get(COL_PV_TEMP), errors="coerce"),
        }
    )
    return out.dropna(subset=["ds"])



def _derive_power_from_energy(df: pd.DataFrame) -> pd.Series:
    if df.empty or "energy_kwh" not in df.columns:
        return pd.Series(pd.NA, index=df.index, dtype="float64")

    work = df.sort_values("ds").copy()
    day_key = work["ds"].dt.date
    delta_kwh = work.groupby(day_key)["energy_kwh"].diff()
    delta_hours = work.groupby(day_key)["ds"].diff().dt.total_seconds() / 3600.0
    derived_kw = delta_kwh / delta_hours

    max_valid_kw = POWER_UPPER_BAD_MW * 1000.0
    valid_mask = (delta_hours > 0) & (delta_kwh >= 0) & (derived_kw <= max_valid_kw)
    result = pd.Series(pd.NA, index=work.index, dtype="float64")
    result.loc[valid_mask] = derived_kw.loc[valid_mask]
    return result.reindex(df.index)



def _shift_ds_hours(df: pd.DataFrame, hours: int) -> pd.DataFrame:
    if df.empty or "ds" not in df.columns or not hours:
        return df

    out = df.copy()
    out["ds"] = pd.to_datetime(out["ds"], errors="coerce") + pd.Timedelta(hours=hours)
    out = out.dropna(subset=["ds"]).copy()
    out["ds"] = out["ds"].dt.floor("h")
    return out


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
    df = _shift_ds_hours(df, HISTORY_SHIFT_HOURS)
    df = df.sort_values("ds").drop_duplicates(subset=["ds"], keep="last").reset_index(drop=True)

    df.loc[(df["power_mw"] < 0) | (df["power_mw"] > POWER_UPPER_BAD_MW), "power_mw"] = pd.NA
    df.loc[(df["irradiation"] < 0) | (df["irradiation"] > IRR_MAX), "irradiation"] = pd.NA
    df.loc[(df["pv_temp"] < PV_TEMP_MIN) | (df["pv_temp"] > PV_TEMP_MAX), "pv_temp"] = pd.NA

    power_from_mw_kw = pd.to_numeric(df["power_mw"], errors="coerce") * 1000.0
    power_from_energy_kw = _derive_power_from_energy(df)

    # Для дней с корректной накопленной выработкой (energy_kwh) используем
    # производную мощность как основную, чтобы дневная энергия сходилась с Excel.
    day_key = pd.to_datetime(df["ds"]).dt.date
    has_energy_day = day_key.map(df.groupby(day_key)["energy_kwh"].apply(lambda x: x.notna().sum() >= 2))

    df["power_kw"] = power_from_mw_kw
    df.loc[has_energy_day, "power_kw"] = power_from_energy_kw.loc[has_energy_day].fillna(
        power_from_mw_kw.loc[has_energy_day]
    )

    hourly = (
        df.set_index("ds")[["irradiation", "air_temp", "pv_temp", "power_kw"]]
        .resample("h")
        .mean()
        .reset_index()
    )
    hourly = hourly[hourly["power_kw"].fillna(0) > MIN_POWER_KW].copy()
    hourly = _align_hourly_day_energy(hourly, daily_targets_kwh)
    return hourly[["ds", "irradiation", "air_temp", "pv_temp", "power_kw"]].reset_index(drop=True)
