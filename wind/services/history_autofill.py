"""Автообновление истории ветростанций из папки/кастомного скрипта."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import logging
import re
from datetime import timedelta
from pathlib import Path

import pandas as pd
from django.utils import timezone

from stations.models import Station
from wind.models import WindRecord

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["ds", "power_kw"]
OPTIONAL_COLUMNS = ["wind_speed_ms", "wind_direction_deg", "air_temp", "air_density"]


def _to_aware_dt(value: pd.Timestamp):
    py_dt = value.to_pydatetime()
    if timezone.is_naive(py_dt):
        return timezone.make_aware(py_dt, timezone.get_current_timezone())
    return timezone.localtime(py_dt)


def _station_data_shift_hours(station: Station) -> int:
    try:
        return int(getattr(station, "data_shift_hours", 0) or 0)
    except (TypeError, ValueError):
        return 0


def normalize_history_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Приводит сырую историю к колонкам WindRecord."""
    col_map = {str(col).strip().lower(): col for col in df.columns}
    ds_col = col_map.get("ds") or col_map.get("timestamp")
    power_col = col_map.get("power_kw") or col_map.get("y")

    if not ds_col or not power_col:
        raise ValueError(f"Нужны колонки ds/timestamp и power_kw/y. Найдено: {list(df.columns)}")

    out = pd.DataFrame(
        {
            "ds": pd.to_datetime(df[ds_col], errors="coerce"),
            "power_kw": pd.to_numeric(df[power_col], errors="coerce"),
        }
    )

    aliases = {
        "wind_speed_ms": ["wind_speed_ms", "wind_speed", "ws"],
        "wind_direction_deg": ["wind_direction_deg", "wind_direction", "wd"],
        "air_temp": ["air_temp", "temperature", "temp"],
        "air_density": ["air_density", "density"],
    }
    for target, candidates in aliases.items():
        source_col = None
        for alias in candidates:
            source_col = col_map.get(alias)
            if source_col:
                break
        if source_col:
            out[target] = pd.to_numeric(df[source_col], errors="coerce")

    out = out.dropna(subset=["ds"]).copy()
    out["ds"] = out["ds"].dt.floor("h")
    out = out.sort_values("ds").drop_duplicates(subset=["ds"], keep="last").reset_index(drop=True)
    return out


def _collect_standard_history_dataframe(folder: Path) -> pd.DataFrame:
    files = [p for p in sorted(folder.glob("*.csv")) if p.is_file()]
    files += [p for p in sorted(folder.glob("*.xlsx")) if p.is_file() and not p.name.startswith("~$")]
    if not files:
        return pd.DataFrame(columns=REQUIRED_COLUMNS + OPTIONAL_COLUMNS)

    parts: list[pd.DataFrame] = []
    for file_path in files:
        try:
            if file_path.suffix.lower() == ".csv":
                raw = pd.read_csv(file_path, low_memory=False)
            else:
                raw = pd.read_excel(file_path)
            normalized = normalize_history_dataframe(raw)
            if not normalized.empty:
                parts.append(normalized)
        except Exception:
            logger.debug("Wind auto-history skip file=%s", file_path, exc_info=True)
            continue

    if not parts:
        return pd.DataFrame(columns=REQUIRED_COLUMNS + OPTIONAL_COLUMNS)
    return pd.concat(parts, ignore_index=True).sort_values("ds").drop_duplicates(subset=["ds"], keep="last")


def _normalize_auto_history_script(raw_value: str) -> str:
    value = (raw_value or "").strip()
    if not value:
        return ""
    for sep in (" или ", " or "):
        if sep in value:
            value = value.split(sep, 1)[0].strip()
            break
    if ":" not in value and "/" not in value and "\\" not in value:
        dot_parts = value.split(".")
        is_module_path = len(dot_parts) > 1 and all(part.isidentifier() for part in dot_parts)
        if not is_module_path:
            value = re.sub(r"[^0-9a-zA-Z_]+", "_", value).strip("_")
    return value


def _load_module_from_file(file_path: str):
    file_obj = Path(file_path)
    if not file_obj.exists() or file_obj.suffix.lower() != ".py":
        raise ValueError(f"Custom history file not found or not a .py file: {file_path}")
    module_name = f"custom_wind_history_{hashlib.md5(str(file_obj).encode('utf-8')).hexdigest()}"
    spec = importlib.util.spec_from_file_location(module_name, str(file_obj))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from file: {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_station_history_builder(station: Station):
    raw_value = _normalize_auto_history_script(getattr(station, "auto_history_script", ""))
    if not raw_value:
        return None

    if ":" in raw_value:
        module_target, _, attr_name = raw_value.partition(":")
    else:
        module_target = raw_value if "." in raw_value else f"wind.services.history_scripts.{raw_value}"
        attr_name = "build_history_dataframe"

    try:
        if module_target.endswith(".py") or "/" in module_target or "\\" in module_target:
            module = _load_module_from_file(module_target.lstrip("/"))
        else:
            module = importlib.import_module(module_target)
        builder = getattr(module, attr_name)
    except Exception:
        logger.exception(
            "Cannot load wind auto_history_script for station_id=%s value=%s",
            station.pk,
            raw_value,
        )
        return None

    if not callable(builder):
        raise TypeError(f"Custom wind history builder is not callable: {raw_value}")
    return builder


def upsert_station_history_from_share(station: Station) -> int:
    folder = Path(getattr(station, "auto_history_folder", "") or "/mnt/share/wind")
    if not folder.exists():
        return 0

    custom_builder = _load_station_history_builder(station)
    if custom_builder is not None:
        df = custom_builder(station)
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Custom wind auto-history builder must return pandas.DataFrame")
        df = normalize_history_dataframe(df)
    else:
        df = _collect_standard_history_dataframe(folder)
    if df.empty:
        return 0

    data_shift_hours = _station_data_shift_hours(station)
    if data_shift_hours:
        df = df.copy()
        df["ds"] = pd.to_datetime(df["ds"], errors="coerce") + timedelta(hours=data_shift_hours)
        df = df.dropna(subset=["ds"])

    ts_values = [_to_aware_dt(ts) for ts in df["ds"]]
    existing_qs = WindRecord.objects.filter(
        station=station,
        history_scope=WindRecord.HISTORY_SCOPE_MAIN,
        timestamp__in=ts_values,
    )
    existing_map = {obj.timestamp: obj for obj in existing_qs}

    create_objs = []
    update_objs = []
    for row in df.itertuples(index=False):
        ts = _to_aware_dt(row.ds)
        obj = existing_map.get(ts)
        payload = {
            "power_kw": float(row.power_kw) if pd.notna(row.power_kw) else None,
            "wind_speed_ms": float(getattr(row, "wind_speed_ms", None)) if pd.notna(getattr(row, "wind_speed_ms", None)) else None,
            "wind_direction_deg": float(getattr(row, "wind_direction_deg", None)) if pd.notna(getattr(row, "wind_direction_deg", None)) else None,
            "air_temp": float(getattr(row, "air_temp", None)) if pd.notna(getattr(row, "air_temp", None)) else None,
            "air_density": float(getattr(row, "air_density", None)) if pd.notna(getattr(row, "air_density", None)) else None,
        }
        if obj is None:
            create_objs.append(
                WindRecord(
                    station=station,
                    history_scope=WindRecord.HISTORY_SCOPE_MAIN,
                    timestamp=ts,
                    **payload,
                )
            )
        else:
            for key, value in payload.items():
                setattr(obj, key, value)
            update_objs.append(obj)

    if create_objs:
        WindRecord.objects.bulk_create(create_objs, batch_size=1000)
    if update_objs:
        WindRecord.objects.bulk_update(
            update_objs,
            ["power_kw", "wind_speed_ms", "wind_direction_deg", "air_temp", "air_density"],
            batch_size=1000,
        )
    return len(create_objs) + len(update_objs)
