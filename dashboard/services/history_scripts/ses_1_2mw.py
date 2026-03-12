from __future__ import annotations

from pathlib import Path

import pandas as pd

MIN_POWER_KW = 0.0001



def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["ds", "irradiation", "air_temp", "pv_temp", "power_kw"])



def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {str(c).strip().lower(): c for c in df.columns}

    ds_col = col_map.get("ds") or col_map.get("timestamp")
    irr_col = col_map.get("irradiation")
    air_col = col_map.get("air_temp")
    pv_col = col_map.get("pv_temp")
    power_col = col_map.get("power_kw")

    required = [ds_col, irr_col, air_col, pv_col, power_col]
    if not all(required):
        return _empty_df()

    out = df[[ds_col, irr_col, air_col, pv_col, power_col]].copy()
    out.columns = ["ds", "irradiation", "air_temp", "pv_temp", "power_kw"]
    out["ds"] = pd.to_datetime(out["ds"], errors="coerce")

    for c in ["irradiation", "air_temp", "pv_temp", "power_kw"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=["ds"]).copy()
    out["ds"] = out["ds"].dt.floor("h")
    return out



def _read_csv_flexible(file_path: Path) -> pd.DataFrame:
    read_attempts = (
        {"low_memory": False},
        {"low_memory": False, "sep": ";", "decimal": ","},
        {"low_memory": False, "sep": ";"},
    )

    for kwargs in read_attempts:
        try:
            return pd.read_csv(file_path, **kwargs)
        except Exception:
            continue

    return pd.DataFrame()


def build_history_dataframe(station) -> pd.DataFrame:
    folder = Path(getattr(station, "auto_history_folder", "") or "")
    if not folder.exists():
        return _empty_df()

    # Для папок с парами meteo(D222*.csv.gz) + отчёт станции (*.xlsx)
    # используем проверенную merge-логику общего обработчика.
    meteo_files = [p for p in sorted(folder.glob("D222*.csv.gz")) if p.is_file()]
    report_files = [p for p in sorted(folder.glob("*.xlsx")) if p.is_file() and not p.name.startswith("~$")]
    if meteo_files and report_files:
        try:
            from dashboard.services.history_autofill import collect_share_history_dataframe

            merged = collect_share_history_dataframe(folder)
            if not merged.empty:
                return merged[["ds", "irradiation", "air_temp", "pv_temp", "power_kw"]].reset_index(drop=True)
        except Exception:
            pass

    files = [p for p in sorted(folder.glob("*.csv")) if p.is_file()]
    files += [p for p in sorted(folder.glob("*.csv.gz")) if p.is_file()]
    files += [p for p in sorted(folder.glob("*.xlsx")) if p.is_file() and not p.name.startswith("~$")]
    if not files:
        return _empty_df()

    parts: list[pd.DataFrame] = []
    for file_path in files:
        try:
            if file_path.suffix.lower() == ".csv" or file_path.name.lower().endswith(".csv.gz"):
                raw = _read_csv_flexible(file_path)
            else:
                raw = pd.read_excel(file_path)
        except Exception:
            continue

        normalized = _normalize_columns(raw)
        if not normalized.empty:
            parts.append(normalized)

    if not parts:
        return _empty_df()

    out = pd.concat(parts, ignore_index=True).sort_values("ds").reset_index(drop=True)
    out = out.drop_duplicates(subset=["ds"], keep="last")
    out = out[out["power_kw"].fillna(0) > MIN_POWER_KW].copy()

    out["irradiation"] = out["irradiation"].round(3)
    out["air_temp"] = out["air_temp"].round(3)
    out["pv_temp"] = out["pv_temp"].round(3)
    out["power_kw"] = out["power_kw"].round(2)

    return out[["ds", "irradiation", "air_temp", "pv_temp", "power_kw"]].reset_index(drop=True)
