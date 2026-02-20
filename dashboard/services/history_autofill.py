from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd
from django.utils import timezone

from solar.models import SolarRecord
from stations.models import Station

MIN_POWER_KW = 0.0001
ROUND_IRR = 3
ROUND_TEMP = 3
ROUND_POWER = 2

logger = logging.getLogger(__name__)


def extract_date_yyyymmdd_from_name(name: str) -> Optional[str]:
    m = re.search(r"(20\d{2})(\d{2})(\d{2})", name)
    if m:
        return f"{m.group(1)}{m.group(2)}{m.group(3)}"

    m = re.search(r"(\d{2})-(\d{2})-(20\d{2})", name)
    if m:
        dd, mm, yyyy = m.group(1), m.group(2), m.group(3)
        return f"{yyyy}{mm}{dd}"

    m = re.search(r"(\d{2})\.(\d{2})\.(20\d{2})", name)
    if m:
        dd, mm, yyyy = m.group(1), m.group(2), m.group(3)
        return f"{yyyy}{mm}{dd}"

    return None


def find_col_by_candidates(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    cols = list(df.columns)
    cols_lc = [str(c).strip().lower() for c in cols]

    for cand in candidates:
        cand_lc = cand.strip().lower()
        for c, lc in zip(cols, cols_lc):
            if lc == cand_lc:
                return c

    for cand in candidates:
        cand_lc = cand.strip().lower()
        for c, lc in zip(cols, cols_lc):
            if cand_lc in lc:
                return c

    return None


def read_meteo_hourly(csv_gz_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_gz_path, low_memory=False)

    time_col = find_col_by_candidates(df, ["time", "datetime", "date_time", "timestamp", "date"])
    if time_col is None:
        time_col = df.columns[0]

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce", dayfirst=False)
    df = df.dropna(subset=[time_col]).copy()

    irr_col = find_col_by_candidates(df, ["irradiation", "irradiance", "ghi", "solar", "radiation", "w/m2", "wm2"])
    air_col = find_col_by_candidates(df, ["air_temp", "air temperature", "temp_air", "tair", "ta", "temperature"])
    pv_col = find_col_by_candidates(df, ["pv_temp", "pv temperature", "module_temp", "panel_temp", "tmodule", "tm"])

    if irr_col is None or air_col is None or pv_col is None:
        raise ValueError(
            f"[METEO] Не нашёл нужные колонки в {csv_gz_path.name}. "
            f"Надо irradiation/air_temp/pv_temp, найдено: {list(df.columns)}"
        )

    for c in [irr_col, air_col, pv_col]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["ds"] = df[time_col].dt.floor("h")

    return (
        df.groupby("ds", as_index=False)
        .agg({irr_col: "mean", air_col: "mean", pv_col: "mean"})
        .rename(columns={irr_col: "irradiation", air_col: "air_temp", pv_col: "pv_temp"})
        .sort_values("ds")
        .reset_index(drop=True)
    )


def read_plant_report_hourly(xlsx_path: Path) -> pd.DataFrame:
    xls = pd.ExcelFile(xlsx_path)

    for sheet in xls.sheet_names:
        raw = pd.read_excel(xlsx_path, sheet_name=sheet, header=None)
        if raw is None or raw.empty:
            continue

        header_row = None
        for i in range(min(120, len(raw))):
            row_vals = raw.iloc[i].astype(str).str.lower().tolist()
            if any("statistical period" in v for v in row_vals) and any("pv yield" in v for v in row_vals):
                header_row = i
                break

        if header_row is None:
            continue

        headers = raw.iloc[header_row].tolist()
        df = raw.iloc[header_row + 1 :].copy()
        df.columns = headers
        df = df.dropna(how="all").copy()

        time_col = find_col_by_candidates(df, ["statistical period", "time", "date", "datetime"])
        kwh_col = find_col_by_candidates(df, ["pv yield (kwh)", "pv yield", "yield (kwh)", "energy (kwh)", "kwh"])

        if time_col is None or kwh_col is None:
            continue

        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df[kwh_col] = pd.to_numeric(df[kwh_col], errors="coerce")
        df = df.dropna(subset=[time_col]).copy()
        df["ds"] = df[time_col].dt.floor("h")

        return (
            df.groupby("ds", as_index=False)[kwh_col]
            .sum()
            .rename(columns={kwh_col: "power_kw"})
            .sort_values("ds")
            .reset_index(drop=True)
        )

    raise ValueError(f"[PLANT] Не нашёл таблицу с 'Statistical Period' и 'PV Yield' в {xlsx_path.name}.")


def is_fusionsolar_report_xlsx(p: Path) -> bool:
    n = p.name.lower()
    if not n.endswith(".xlsx") or n.startswith("~$"):
        return False
    return any(
        marker in n
        for marker in [
            "plant report",
            "plant statistics",
            "statistics report_by time",
        ]
    ) or n.startswith("reportspp")


def pick_best_report_for_date(files: list[Path]) -> Path:
    if len(files) == 1:
        return files[0]
    plant_report = [f for f in files if "plant report" in f.name.lower()]
    if plant_report:
        return max(plant_report, key=lambda x: x.stat().st_mtime)
    return max(files, key=lambda x: x.stat().st_mtime)


def _clean_round_filter(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["irradiation", "air_temp", "pv_temp", "power_kw"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df[df["power_kw"].fillna(0) > MIN_POWER_KW].copy()
    df["irradiation"] = df["irradiation"].round(ROUND_IRR)
    df["air_temp"] = df["air_temp"].round(ROUND_TEMP)
    df["pv_temp"] = df["pv_temp"].round(ROUND_TEMP)
    df["power_kw"] = df["power_kw"].round(ROUND_POWER)
    return df


def _to_aware_dt(value: pd.Timestamp):
    py_dt = value.to_pydatetime()
    if timezone.is_naive(py_dt):
        return timezone.make_aware(py_dt, timezone.get_current_timezone())
    return timezone.localtime(py_dt)


def _merge_one_day(meteo_hourly: pd.DataFrame, plant_hourly: pd.DataFrame) -> pd.DataFrame:
    return pd.merge(meteo_hourly, plant_hourly, on="ds", how="inner")[
        ["ds", "irradiation", "air_temp", "pv_temp", "power_kw"]
    ].copy()


def collect_share_history_dataframe(folder: Path) -> pd.DataFrame:
    meteo_files = sorted(folder.glob("D222*.csv.gz"))
    plant_files = [p for p in sorted(folder.glob("*.xlsx")) if is_fusionsolar_report_xlsx(p)]

    if not meteo_files or not plant_files:
        return pd.DataFrame(columns=["ds", "irradiation", "air_temp", "pv_temp", "power_kw"])

    plant_by_date_multi: dict[str, list[Path]] = {}
    for p in plant_files:
        d = extract_date_yyyymmdd_from_name(p.name)
        if d:
            plant_by_date_multi.setdefault(d, []).append(p)

    plant_by_date = {d: pick_best_report_for_date(lst) for d, lst in plant_by_date_multi.items()}

    meteo_by_date: dict[str, Path] = {}
    for m in meteo_files:
        d = extract_date_yyyymmdd_from_name(m.name)
        if d:
            meteo_by_date[d] = m

    common_dates = sorted(set(plant_by_date.keys()) & set(meteo_by_date.keys()))
    if not common_dates:
        return pd.DataFrame(columns=["ds", "irradiation", "air_temp", "pv_temp", "power_kw"])

    rows: list[pd.DataFrame] = []
    for d in common_dates:
        try:
            meteo_hourly = read_meteo_hourly(meteo_by_date[d])
            plant_hourly = read_plant_report_hourly(plant_by_date[d])
            rows.append(_merge_one_day(meteo_hourly, plant_hourly))
        except Exception:
            logger.exception("Auto-history: skip date=%s folder=%s", d, folder)

    if not rows:
        return pd.DataFrame(columns=["ds", "irradiation", "air_temp", "pv_temp", "power_kw"])

    out = pd.concat(rows, ignore_index=True).sort_values("ds").reset_index(drop=True)
    out = out.drop_duplicates(subset=["ds"], keep="last")
    return _clean_round_filter(out)


def upsert_station_history_from_share(station: Station) -> int:
    folder = Path(station.auto_history_folder or "/mnt/share")
    if not folder.exists():
        return 0

    df = collect_share_history_dataframe(folder)
    if df.empty:
        return 0

    ts_values = [_to_aware_dt(ts) for ts in df["ds"]]
    existing_qs = SolarRecord.objects.filter(
        station=station,
        history_scope=SolarRecord.HISTORY_SCOPE_MAIN,
        timestamp__in=ts_values,
    )
    existing_map = {obj.timestamp: obj for obj in existing_qs}

    create_objs = []
    update_objs = []

    for row in df.itertuples(index=False):
        ts = _to_aware_dt(row.ds)
        obj = existing_map.get(ts)
        if obj is None:
            create_objs.append(
                SolarRecord(
                    station=station,
                    history_scope=SolarRecord.HISTORY_SCOPE_MAIN,
                    timestamp=ts,
                    power_kw=float(row.power_kw) if pd.notna(row.power_kw) else None,
                    irradiation=float(row.irradiation) if pd.notna(row.irradiation) else None,
                    air_temp=float(row.air_temp) if pd.notna(row.air_temp) else None,
                    pv_temp=float(row.pv_temp) if pd.notna(row.pv_temp) else None,
                )
            )
            continue

        obj.power_kw = float(row.power_kw) if pd.notna(row.power_kw) else None
        obj.irradiation = float(row.irradiation) if pd.notna(row.irradiation) else None
        obj.air_temp = float(row.air_temp) if pd.notna(row.air_temp) else None
        obj.pv_temp = float(row.pv_temp) if pd.notna(row.pv_temp) else None
        update_objs.append(obj)

    if create_objs:
        SolarRecord.objects.bulk_create(create_objs, batch_size=1000)
    if update_objs:
        SolarRecord.objects.bulk_update(update_objs, ["power_kw", "irradiation", "air_temp", "pv_temp"], batch_size=1000)

    return len(create_objs) + len(update_objs)


def run_auto_history_updates() -> int:
    updated_rows = 0
    for station in Station.objects.filter(auto_history_enabled=True):
        try:
            updated_rows += upsert_station_history_from_share(station)
        except Exception:
            logger.exception(
                "Auto-history failed for station_id=%s name=%s folder=%s",
                station.pk,
                station.name,
                station.auto_history_folder,
            )
    return updated_rows
