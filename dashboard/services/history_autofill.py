from __future__ import annotations

import hashlib
import importlib
import importlib.util
import logging
import re
from datetime import timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from django.utils import timezone

from solar.models import SolarRecord
from solar.org_sync import sync_solar_records
from stations.models import Station

MIN_POWER_KW = 0.0001
ROUND_IRR = 3
ROUND_TEMP = 3
ROUND_POWER = 2

logger = logging.getLogger(__name__)
EARLY_FALLBACK_WINDOW_MINUTES = 120


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


def _normalize_standard_history_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {str(col).strip().lower(): col for col in df.columns}

    ds_col = col_map.get("ds") or col_map.get("timestamp")
    irr_col = col_map.get("irradiation")
    air_col = col_map.get("air_temp")
    pv_col = col_map.get("pv_temp")
    power_col = col_map.get("power_kw")

    if not all([ds_col, irr_col, air_col, pv_col, power_col]):
        raise ValueError(
            "[STANDARD] Нужны колонки ds, Irradiation, Air_Temp, PV_Temp, Power_KW "
            f"(или timestamp вместо ds). Найдено: {list(df.columns)}"
        )

    out = df[[ds_col, irr_col, air_col, pv_col, power_col]].copy()
    out.columns = ["ds", "irradiation", "air_temp", "pv_temp", "power_kw"]
    out["ds"] = pd.to_datetime(out["ds"], errors="coerce")
    out = out.dropna(subset=["ds"]).copy()
    out["ds"] = out["ds"].dt.floor("h")
    return out


def _collect_standard_history_dataframe(folder: Path) -> pd.DataFrame:
    files = [p for p in sorted(folder.glob("*.csv")) if p.is_file()]
    files += [p for p in sorted(folder.glob("*.xlsx")) if p.is_file() and not p.name.startswith("~$")]
    if not files:
        return pd.DataFrame(columns=["ds", "irradiation", "air_temp", "pv_temp", "power_kw"])

    parts: list[pd.DataFrame] = []
    for file_path in files:
        try:
            if file_path.suffix.lower() == ".csv":
                raw = pd.read_csv(file_path, low_memory=False)
            else:
                raw = pd.read_excel(file_path)
            normalized = _normalize_standard_history_columns(raw)
            if not normalized.empty:
                parts.append(normalized)
        except Exception:
            logger.debug("Auto-history: skip standard file=%s", file_path, exc_info=True)
            continue

    if not parts:
        return pd.DataFrame(columns=["ds", "irradiation", "air_temp", "pv_temp", "power_kw"])

    out = pd.concat(parts, ignore_index=True).sort_values("ds").reset_index(drop=True)
    out = out.drop_duplicates(subset=["ds"], keep="last")
    return _clean_round_filter(out)


def _to_aware_dt(value: pd.Timestamp):
    py_dt = value.to_pydatetime()
    if timezone.is_naive(py_dt):
        return timezone.make_aware(py_dt, timezone.get_current_timezone())
    return timezone.localtime(py_dt)


def _merge_one_day(meteo_hourly: pd.DataFrame, plant_hourly: pd.DataFrame) -> pd.DataFrame:
    return pd.merge(meteo_hourly, plant_hourly, on="ds", how="inner")[
        ["ds", "irradiation", "air_temp", "pv_temp", "power_kw"]
    ].copy()


def _merge_plant_reports_for_day(plant_files: list[Path]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for pf in plant_files:
        hourly = read_plant_report_hourly(pf)
        if not hourly.empty:
            rows.append(hourly)

    if not rows:
        return pd.DataFrame(columns=["ds", "power_kw"])

    merged = pd.concat(rows, ignore_index=True)
    merged["power_kw"] = pd.to_numeric(merged["power_kw"], errors="coerce")
    return (
        merged.groupby("ds", as_index=False)["power_kw"]
        .sum(min_count=1)
        .sort_values("ds")
        .reset_index(drop=True)
    )


def _safe_collect_day(folder: Path, date_key: str, meteo_file: Path, plant_files: list[Path]) -> Optional[pd.DataFrame]:
    try:
        meteo_hourly = read_meteo_hourly(meteo_file)
        plant_hourly = _merge_plant_reports_for_day(plant_files)
        return _merge_one_day(meteo_hourly, plant_hourly)
    except Exception:
        logger.exception("Auto-history: skip date=%s folder=%s", date_key, folder)
        return None


def collect_share_history_dataframe(folder: Path) -> pd.DataFrame:
    meteo_files = sorted(folder.glob("D222*.csv.gz"))
    plant_files = [p for p in sorted(folder.glob("*.xlsx")) if is_fusionsolar_report_xlsx(p)]

    if not meteo_files or not plant_files:
        return _collect_standard_history_dataframe(folder)

    plant_by_date_multi: dict[str, list[Path]] = {}
    for p in plant_files:
        d = extract_date_yyyymmdd_from_name(p.name)
        if d:
            plant_by_date_multi.setdefault(d, []).append(p)

    plant_by_date = plant_by_date_multi

    meteo_by_date: dict[str, Path] = {}
    for m in meteo_files:
        d = extract_date_yyyymmdd_from_name(m.name)
        if d:
            meteo_by_date[d] = m

    common_dates = sorted(set(plant_by_date.keys()) & set(meteo_by_date.keys()))
    if not common_dates:
        return _collect_standard_history_dataframe(folder)

    rows: list[pd.DataFrame] = []
    for d in common_dates:
        day_df = _safe_collect_day(folder, d, meteo_by_date[d], plant_by_date[d])
        if day_df is not None:
            rows.append(day_df)

    if not rows:
        return _collect_standard_history_dataframe(folder)

    out = pd.concat(rows, ignore_index=True).sort_values("ds").reset_index(drop=True)
    out = out.drop_duplicates(subset=["ds"], keep="last")
    return _clean_round_filter(out)


def _load_module_from_file(file_path: str):
    file_obj = Path(file_path)
    if not file_obj.exists() or file_obj.suffix.lower() != ".py":
        raise ValueError(f"Custom history file not found or not a .py file: {file_path}")

    module_name = f"custom_history_{hashlib.md5(str(file_obj).encode('utf-8')).hexdigest()}"
    spec = importlib.util.spec_from_file_location(module_name, str(file_obj))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from file: {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _normalize_auto_history_script(raw_value: str) -> str:
    value = (raw_value or "").strip()
    if not value:
        return ""

    for sep in (" или ", " or "):
        if sep in value:
            first = value.split(sep, 1)[0].strip()
            if first:
                value = first
            break

    has_func_delimiter = ":" in value
    has_file_suffix = value.lower().endswith(".py")

    # Поддержка UI-вариантов с путями (например /history_scripts/ses_8_8mw
    # или dashboard/services/history_scripts/ses_8_8mw).
    # Для не-.py путей берём последний токен как короткое имя модуля.
    if not has_func_delimiter and ("/" in value or "\\" in value):
        tokens = [t for t in re.split(r"[\\/]+", value) if t]
        if tokens and not has_file_suffix:
            value = tokens[-1]

    # Поддержка "человеческого" ввода в UI: "ses 8 8mw" -> "ses_8_8mw"
    # Нормализацию применяем только к короткому имени модуля,
    # не затрагивая полные python-пути и file.py:func.
    if ":" not in value and "." not in value and "/" not in value and "\\" not in value:
        value = "_".join(value.split())

    return value


def _load_station_history_builder(station: Station):
    raw_value = _normalize_auto_history_script(getattr(station, "auto_history_script", ""))
    if not raw_value:
        return None

    if ":" in raw_value:
        module_target, _, attr_name = raw_value.partition(":")
    else:
        # Поддержка значений без ':function':
        # - короткое имя: ses_50_balkhash
        # - модульный путь: dashboard.services.history_scripts.ses_50_balkhash
        # - путь к файлу: /dashboard/services/history_scripts/ses_50_balkhash.py
        if raw_value.endswith(".py") or "/" in raw_value or "\\" in raw_value:
            module_target = raw_value
        elif "." in raw_value:
            module_target = raw_value
        else:
            module_target = f"dashboard.services.history_scripts.{raw_value}"
        attr_name = "build_history_dataframe"

    module_target = module_target.strip()
    attr_name = attr_name.strip()

    if not module_target or not attr_name:
        logger.warning(
            "Invalid auto_history_script format for station_id=%s: %s. Fallback to standard handler.",
            station.pk,
            raw_value,
        )
        return None

    try:
        if module_target.endswith('.py') or '/' in module_target:
            module = None
            file_candidate = module_target

            # UI часто сохраняет путь как "/dashboard/services/history_scripts/...py:func"
            # (абсолютный от корня ФС), хотя в проекте файл лежит относительно репозитория.
            # Пробуем несколько безопасных вариантов до fallback:
            # 1) как есть;
            # 2) относительный путь без ведущего '/';
            # 3) короткое имя модуля через package import.
            try_paths = [file_candidate]
            if file_candidate.startswith("/"):
                try_paths.append(file_candidate.lstrip("/"))

            for path_candidate in try_paths:
                try:
                    module = _load_module_from_file(path_candidate)
                    break
                except Exception:
                    module = None

            if module is None:
                script_name = Path(file_candidate).stem
                if script_name:
                    module = importlib.import_module(f"dashboard.services.history_scripts.{script_name}")
                else:
                    raise ImportError(f"Cannot resolve auto_history_script path: {module_target}")
        else:
            module = importlib.import_module(module_target)

        builder = getattr(module, attr_name)
    except Exception:
        logger.exception(
            "Cannot load auto_history_script for station_id=%s value=%s. Fallback to standard handler.",
            station.pk,
            raw_value,
        )
        return None

    if not callable(builder):
        raise TypeError(f"Custom history builder is not callable: {raw_value}")
    return builder






def _normalize_folder_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (value or "").lower())


def _resolve_station_folder_alias(station: Station, folder: Path, share_root: Path) -> Optional[Path]:
    org_id = getattr(station, "org_id", None)
    station_name = getattr(station, "name", "")

    org_dir = share_root / f"org_{org_id}" if org_id else folder.parent
    if not org_dir.exists() or not org_dir.is_dir():
        return None

    expected_keys = {
        _normalize_folder_key(folder.name),
        _normalize_folder_key(station_name),
    }
    expected_keys.discard("")
    if not expected_keys:
        return None

    for child in org_dir.iterdir():
        if not child.is_dir():
            continue
        child_key = _normalize_folder_key(child.name)
        if child_key in expected_keys:
            return child

    return None

def _resolve_station_share_folder(station: Station, share_root: Optional[Path] = None) -> Path:
    folder = Path(getattr(station, "auto_history_folder", "") or "/mnt/share")
    if folder.exists():
        return folder

    base = share_root or Path("/mnt/share")
    folder_str = str(folder)
    base_prefix = f"{str(base).rstrip('/')}/"
    if folder_str.startswith(base_prefix) and base.exists():
        alias = _resolve_station_folder_alias(station, folder, base)
        if alias is not None:
            logger.warning(
                "Auto-history folder alias used station_id=%s configured=%s resolved=%s",
                getattr(station, "pk", None),
                folder,
                alias,
            )
            return alias

        logger.warning(
            "Auto-history folder missing for station_id=%s folder=%s, fallback to shared root=%s",
            getattr(station, "pk", None),
            folder,
            base,
        )
        return base

    return folder

def upsert_station_history_from_share(station: Station) -> int:
    folder = _resolve_station_share_folder(station)
    if not folder.exists():
        return 0

    custom_builder = _load_station_history_builder(station)
    if custom_builder is not None:
        df = custom_builder(station)
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Custom auto-history builder must return pandas.DataFrame")
    else:
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

    if ts_values:
        mirrored_qs = SolarRecord.objects.filter(
            station=station,
            history_scope=SolarRecord.HISTORY_SCOPE_MAIN,
            timestamp__in=ts_values,
        ).order_by("id")
        sync_solar_records(mirrored_qs)

    return len(create_objs) + len(update_objs)


def _safe_upsert_station(station: Station) -> tuple[int, bool]:
    try:
        return upsert_station_history_from_share(station), True
    except Exception:
        logger.exception(
            "Auto-history failed for station_id=%s name=%s folder=%s",
            station.pk,
            station.name,
            station.auto_history_folder,
        )
        return 0, False



def _is_station_due_for_auto_history(station: Station, now_local) -> bool:
    last_run_date = getattr(station, "auto_history_last_run_date", None)
    if last_run_date == now_local.date():
        return False

    run_time = getattr(station, "auto_history_run_time", None)
    if run_time is None:
        return True

    now_dt = now_local.replace(second=0, microsecond=0)
    scheduled_dt = now_dt.replace(hour=run_time.hour, minute=run_time.minute)

    # Небольшой grace-период защищает от пропусков на границе времени
    # (например tick в 09:19:31 при расписании 09:20).
    if now_dt + timedelta(minutes=1) >= scheduled_dt:
        return True

    # Fallback для редких запусков планировщика (например, 1 раз в день):
    # если станция уже проверялась в прошлые дни, но сегодня ещё нет,
    # разрешаем выполнить проверку до времени auto_history_run_time,
    # чтобы не ждать целые сутки до следующего тика.
    return last_run_date is not None and last_run_date < now_local.date()


def _mark_station_auto_history_checked(station: Station, check_date):
    if getattr(station, "auto_history_last_run_date", None) == check_date:
        return
    station.auto_history_last_run_date = check_date
    station.save(update_fields=["auto_history_last_run_date"])


def _record_auto_history_tick(station: Station, checked_at, status: str, rows: int, message: str) -> None:
    update_fields = []

    if getattr(station, "auto_history_last_check_at", None) != checked_at:
        station.auto_history_last_check_at = checked_at
        update_fields.append("auto_history_last_check_at")

    if getattr(station, "auto_history_last_status", "") != (status or ""):
        station.auto_history_last_status = status or ""
        update_fields.append("auto_history_last_status")

    rows_int = int(rows or 0)
    if getattr(station, "auto_history_last_rows", 0) != rows_int:
        station.auto_history_last_rows = rows_int
        update_fields.append("auto_history_last_rows")

    if getattr(station, "auto_history_last_message", "") != (message or ""):
        station.auto_history_last_message = message or ""
        update_fields.append("auto_history_last_message")

    if update_fields:
        station.save(update_fields=update_fields)


def run_auto_history_updates() -> int:
    updated_rows = 0
    now_local = timezone.localtime()
    for station in Station.objects.filter(auto_history_enabled=True):
        run_time = getattr(station, "auto_history_run_time", None)
        if not _is_station_due_for_auto_history(station, now_local):
            logger.info(
                "Auto-history skip station_id=%s now=%s run_time=%s last_run_date=%s",
                station.pk,
                now_local.strftime("%Y-%m-%d %H:%M:%S%z"),
                run_time,
                getattr(station, "auto_history_last_run_date", None),
            )
            _record_auto_history_tick(
                station,
                now_local,
                status="skipped",
                rows=0,
                message=f"Skip: now<{run_time} или уже выполнено сегодня",
            )
            continue

        logger.info(
            "Auto-history due station_id=%s now=%s run_time=%s last_run_date=%s",
            station.pk,
            now_local.strftime("%Y-%m-%d %H:%M:%S%z"),
            getattr(station, "auto_history_run_time", None),
            getattr(station, "auto_history_last_run_date", None),
        )

        rows, success = _safe_upsert_station(station)
        updated_rows += rows

        # Помечаем станцию как проверенную за день только если
        # обновление действительно прошло успешно и были изменения.
        # Иначе оставляем возможность повторной проверки в этот же день
        # (например, если файлы появились позже или был временный сбой).
        if success and rows > 0:
            _mark_station_auto_history_checked(station, now_local.date())
            _record_auto_history_tick(
                station,
                now_local,
                status="updated",
                rows=rows,
                message=f"Обновлено строк: {rows}",
            )
            logger.info("Auto-history marked checked station_id=%s rows=%s", station.pk, rows)
        elif success:
            _record_auto_history_tick(
                station,
                now_local,
                status="no_rows",
                rows=0,
                message="Автообновление выполнено, новых строк нет.",
            )
            logger.warning("Auto-history no new rows station_id=%s", station.pk)
        else:
            _record_auto_history_tick(
                station,
                now_local,
                status="failed",
                rows=0,
                message="Ошибка автообновления. См. логи сервера.",
            )
            logger.warning("Auto-history failed station_id=%s", station.pk)

    return updated_rows
