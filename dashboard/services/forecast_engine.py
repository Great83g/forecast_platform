# dashboard/services/forecast_engine.py
from __future__ import annotations

import json
import logging
import importlib.util
import time
from datetime import date, timedelta
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from django.conf import settings
from django.db import OperationalError, transaction
from django.utils import timezone

from neuralprophet import load as np_load

from solar.models import SolarForecast, SolarRecord
from solar.org_sync import sync_solar_forecasts
from stations.models import Station
from .model_storage import resolve_station_model_dir
from .open_meteo import fetch_open_meteo_hourly
from .vc_weather import fetch_visual_crossing_hourly
from .forecast_guardrails import apply_visual_crossing_fallback, write_forecast_guardrail_log


MODEL_DIR: Path = Path(getattr(settings, "MODEL_DIR", Path(settings.BASE_DIR) / "models_cache"))
logger = logging.getLogger(__name__)


def _xgb_module() -> Any:
    if importlib.util.find_spec("xgboost") is None:
        return None
    import xgboost

    return xgboost


def _station_data_shift_hours(station: Station) -> int:
    try:
        return int(getattr(station, "data_shift_hours", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _station_forecast_shift_hours(station: Station) -> int:
    try:
        return int(getattr(station, "forecast_shift_hours", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _station_model_dir(station: Station) -> Path:
    return resolve_station_model_dir(MODEL_DIR, station)


def _model_paths_for_station(station: Station) -> Dict[str, Path]:
    model_dir = _station_model_dir(station)
    return {
        "np": model_dir / "np_model.np",
        "np_meta": model_dir / "np_model.meta.json",
        "xgb": model_dir / "xgb_model.json",
        "xgb_meta": model_dir / "xgb_model.meta.json",
        "legacy_np": MODEL_DIR / f"np_model_{station.pk}.np",
        "legacy_np_meta": MODEL_DIR / f"np_model_{station.pk}.meta.json",
        "legacy_xgb": MODEL_DIR / f"xgb_model_{station.pk}.json",
        "legacy_xgb_meta": MODEL_DIR / f"xgb_model_{station.pk}.meta.json",
    }


def _model_file_is_stale(path: Path, max_age_days: int) -> bool:
    if max_age_days <= 0:
        return False
    if not path.exists():
        return True
    try:
        age_seconds = max(0.0, float(timezone.now().timestamp()) - float(path.stat().st_mtime))
    except Exception:
        return False
    return age_seconds > float(timedelta(days=max_age_days).total_seconds())


XGB_EXPECTED_FEATURES = [
    "Irradiation",
    "Air_Temp",
    "PV_Temp",
    "Wind_Speed",
    "hour",
    "month",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "sun_elev_deg",
    "low_sun_flag",
    "sunrise_hour_flag",
    "solar_ramp_factor",
    "irradiation_x_ramp",
]

PR_FOR_EXPECTED = 0.90
FORECAST_GLOBAL_BIAS_MAX = 1.5
FORECAST_IRRADIATION_NOISE_WM2_DEFAULT = 35.0
FORECAST_MORNING_IRR_BOOST_DEFAULT = 1.08
FORECAST_MORNING_IRR_BOOST_MAX = 1.35
FORECAST_CLEAR_SKY_FLOOR_RATIO_DEFAULT = 0.92
FORECAST_EARLY_MORNING_CAP_UPLIFT_DEFAULT = 1.10
FORECAST_EARLY_MORNING_CAP_MIN_SAMPLES_DEFAULT = 5

AUTO_SNOWDEPTH_M_THRESHOLD = 0.02
AUTO_TEMP_MAX_FOR_SNOW = 2.0
AUTO_SNOW_FACTOR = 1.0
MANUAL_SNOW_FACTOR_MAX = 1.5
AUTO_FOG_FACTOR = 1.0
FOG_CODES = {45, 48}
SNOW_CODES = {71, 73, 75, 77, 85, 86}




def _forecast_db_lock_retries() -> int:
    raw = getattr(settings, "FORECAST_DB_LOCK_RETRIES", 8)
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 8


def _forecast_db_lock_retry_delay_seconds() -> float:
    raw = getattr(settings, "FORECAST_DB_LOCK_RETRY_DELAY_SECONDS", 0.75)
    try:
        return float(np.clip(float(raw), 0.05, 10.0))
    except (TypeError, ValueError):
        return 0.75


def _replace_solar_forecast_rows_with_retry(
    *,
    station: Station,
    forecast_scope: str,
    cleanup_start,
    cleanup_end,
    objs: List[SolarForecast],
) -> None:
    """Replace forecast rows with SQLite lock retries.

    The expensive weather/model prediction work happens before this helper. Only
    the small delete+bulk_create section is wrapped in a transaction, which keeps
    SQLite write locks short and makes manual recalculation less likely to fight
    the scheduler/web process.
    """
    retries = _forecast_db_lock_retries()
    delay = _forecast_db_lock_retry_delay_seconds()

    for attempt in range(retries + 1):
        try:
            with transaction.atomic():
                SolarForecast.objects.filter(
                    station=station,
                    forecast_scope=forecast_scope,
                    timestamp__gte=cleanup_start,
                    timestamp__lt=cleanup_end,
                ).delete()
                SolarForecast.objects.bulk_create(objs, batch_size=500)
            return
        except OperationalError as exc:
            if "locked" not in str(exc).lower() or attempt >= retries:
                raise
            sleep_seconds = delay * (attempt + 1)
            logger.warning(
                "[FORECAST_DB] database locked while saving station=%s scope=%s rows=%s; retry %s/%s in %.2fs",
                station.pk,
                forecast_scope,
                len(objs),
                attempt + 1,
                retries,
                sleep_seconds,
            )
            time.sleep(sleep_seconds)

def _forecast_global_bias() -> float:
    raw = getattr(settings, "FORECAST_GLOBAL_BIAS", 1.0)
    try:
        bias = float(raw)
    except (TypeError, ValueError):
        bias = 1.0
    return float(np.clip(bias, 0.0, FORECAST_GLOBAL_BIAS_MAX))


def _forecast_irradiation_noise_floor_wm2() -> float:
    raw = getattr(settings, "FORECAST_IRRADIATION_NOISE_WM2", FORECAST_IRRADIATION_NOISE_WM2_DEFAULT)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = FORECAST_IRRADIATION_NOISE_WM2_DEFAULT
    return float(np.clip(value, 0.0, 120.0))


def _forecast_morning_irradiation_boost() -> float:
    raw = getattr(settings, "FORECAST_MORNING_IRR_BOOST", FORECAST_MORNING_IRR_BOOST_DEFAULT)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = FORECAST_MORNING_IRR_BOOST_DEFAULT
    return float(np.clip(value, 1.0, FORECAST_MORNING_IRR_BOOST_MAX))


def _forecast_clear_sky_floor_ratio() -> float:
    raw = getattr(settings, "FORECAST_CLEAR_SKY_FLOOR_RATIO", FORECAST_CLEAR_SKY_FLOOR_RATIO_DEFAULT)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = FORECAST_CLEAR_SKY_FLOOR_RATIO_DEFAULT
    return float(np.clip(value, 0.0, 1.0))


def _forecast_early_morning_cap_uplift() -> float:
    raw = getattr(settings, "FORECAST_EARLY_MORNING_CAP_UPLIFT", FORECAST_EARLY_MORNING_CAP_UPLIFT_DEFAULT)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = FORECAST_EARLY_MORNING_CAP_UPLIFT_DEFAULT
    return float(np.clip(value, 1.0, 1.5))


def _forecast_early_morning_cap_min_samples() -> int:
    raw = getattr(settings, "FORECAST_EARLY_MORNING_CAP_MIN_SAMPLES", FORECAST_EARLY_MORNING_CAP_MIN_SAMPLES_DEFAULT)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = FORECAST_EARLY_MORNING_CAP_MIN_SAMPLES_DEFAULT
    return max(1, value)


def _is_single_axis_tracker(st: Station) -> bool:
    mount_type = str(getattr(st, "mount_type", Station.MOUNT_FIXED) or Station.MOUNT_FIXED)
    mount_type = mount_type.strip().lower().replace("-", "_")
    return mount_type == Station.MOUNT_SINGLE_AXIS_TRACKER


def _station_ac_nameplate_mw(st: Station, fallback_mw: float) -> float:
    for field_name in ("capacity_ac_kw", "capacity_kw"):
        raw_value = getattr(st, field_name, None)
        if raw_value:
            try:
                value_mw = float(raw_value) / 1000.0
            except (TypeError, ValueError):
                continue
            if np.isfinite(value_mw) and value_mw > 0:
                return value_mw

    raw_value = getattr(st, "capacity_ac_mw", None)
    if raw_value:
        try:
            value_mw = float(raw_value)
        except (TypeError, ValueError):
            value_mw = 0.0
        if np.isfinite(value_mw) and value_mw > 0:
            return value_mw

    return float(fallback_mw)


def _station_pr_default(st: Station) -> float:
    try:
        value = float(getattr(st, "pr_default", PR_FOR_EXPECTED) or PR_FOR_EXPECTED)
    except (TypeError, ValueError):
        value = PR_FOR_EXPECTED
    return float(np.clip(value, 0.10, 1.00))


def _historical_tracker_output_cap_mw(st: Station, ac_cap_mw: float) -> Optional[float]:
    """
    Safe tracker cap from station history.

    For tracker stations we keep the AC nameplate as the hard ceiling and, when
    enough actual output exists, also respect the station-specific p95/p99 peak
    envelope. A small uplift avoids cutting normal near-record points while still
    blocking post-processing spikes.
    """
    rows = list(
        SolarRecord.objects.filter(
            station=st,
            history_scope=SolarRecord.HISTORY_SCOPE_MAIN,
            power_kw__isnull=False,
        )
        .order_by("-timestamp")
        .values_list("power_kw", flat=True)[:24 * 180]
    )
    if len(rows) < 24:
        return float(ac_cap_mw)

    values_mw = (pd.to_numeric(pd.Series(rows), errors="coerce") / 1000.0).dropna()
    values_mw = values_mw[(values_mw >= 0.0) & np.isfinite(values_mw)]
    if len(values_mw) < 24:
        return float(ac_cap_mw)

    p95 = float(values_mw.quantile(0.95))
    p99 = float(values_mw.quantile(0.99))
    if not np.isfinite(p95) or not np.isfinite(p99) or p99 <= 0:
        return float(ac_cap_mw)

    historical_cap = max(p95 * 1.08, p99 * 1.02)
    return float(np.clip(historical_cap, 0.0, ac_cap_mw))


def _historical_tracker_hourly_profile_mw(st: Station, ac_cap_mw: float) -> Dict[int, Dict[str, float]]:
    """Return clear-day hourly median/p75/p95 AC output profile for tracker shaping."""
    rows = list(
        SolarRecord.objects.filter(
            station=st,
            history_scope=SolarRecord.HISTORY_SCOPE_MAIN,
            power_kw__isnull=False,
        )
        .exclude(irradiation__isnull=True)
        .order_by("-timestamp")
        .values("timestamp", "power_kw", "irradiation")[:24 * 365]
    )
    if len(rows) < 24:
        return {}

    hist = pd.DataFrame(rows)
    hist["timestamp"] = pd.to_datetime(hist["timestamp"], errors="coerce")
    hist["hour"] = hist["timestamp"].dt.hour.astype("Int64")
    hist["date"] = hist["timestamp"].dt.date
    hist["power_mw"] = (
        pd.to_numeric(hist["power_kw"], errors="coerce") / 1000.0
    ).clip(0.0, ac_cap_mw)
    hist["irradiation"] = pd.to_numeric(hist["irradiation"], errors="coerce")
    hist = hist.dropna(subset=["hour", "date", "power_mw", "irradiation"])
    if len(hist) < 24:
        return {}

    clear_window = hist[hist["hour"].between(8, 15)]
    if clear_window.empty:
        return {}

    day_stats = clear_window.groupby("date").agg(
        rows=("power_mw", "count"),
        mean_irr=("irradiation", "mean"),
        max_irr=("irradiation", "max"),
    )
    clear_dates = set(
        day_stats.loc[
            (day_stats["rows"] >= 4)
            & (day_stats["mean_irr"] >= 450.0)
            & (day_stats["max_irr"] >= 650.0)
        ].index
    )

    if clear_dates:
        clear_hist = hist[
            hist["date"].isin(clear_dates)
            & hist["hour"].between(6, 19)
            & (hist["irradiation"] >= 80.0)
        ].copy()
    else:
        clear_hist = hist[hist["hour"].between(6, 19) & (hist["irradiation"] >= 500.0)].copy()

    if len(clear_hist) < 12:
        return {}

    profile: Dict[int, Dict[str, float]] = {}
    for hour, values in clear_hist.groupby("hour")["power_mw"]:
        values = values.dropna()
        if len(values) < 2:
            continue
        profile[int(hour)] = {
            "median": float(values.quantile(0.50)),
            "p75": float(values.quantile(0.75)),
            "p95": float(values.quantile(0.95)),
            "samples": float(len(values)),
        }
    return profile


def _apply_tracker_midday_expected_floor(
    y_final: np.ndarray,
    feat: pd.DataFrame,
    st: Station,
    ac_cap_mw: float,
) -> np.ndarray:
    """
    Tracker midday guardrail (all single-axis tracker stations):
    if Irradiation >= 750 and hour in [09..16], forecast cannot be below
    historical expected output by (hour, irradiation-bin), capped by AC.
    """
    if not _is_single_axis_tracker(st):
        return y_final

    rows = list(
        SolarRecord.objects.filter(
            station=st,
            history_scope=SolarRecord.HISTORY_SCOPE_MAIN,
            power_kw__isnull=False,
            irradiation__isnull=False,
        )
        .order_by("-timestamp")
        .values("timestamp", "power_kw", "irradiation")[:24 * 365]
    )
    if len(rows) < 72:
        return y_final

    hist = pd.DataFrame(rows)
    hist["timestamp"] = pd.to_datetime(hist["timestamp"], errors="coerce")
    hist["hour"] = hist["timestamp"].dt.hour
    hist["irr"] = pd.to_numeric(hist["irradiation"], errors="coerce")
    hist["power_mw"] = pd.to_numeric(hist["power_kw"], errors="coerce") / 1000.0
    hist = hist.dropna(subset=["hour", "irr", "power_mw"])
    if hist.empty:
        return y_final

    hist = hist[(hist["hour"] >= 9) & (hist["hour"] <= 16) & (hist["irr"] >= 0)]
    if hist.empty:
        return y_final

    bin_edges = [0, 250, 500, 750, 900, 1100, 2000]
    hist["irr_bin"] = pd.cut(hist["irr"], bins=bin_edges, include_lowest=True, right=False)
    grouped = (
        hist.groupby(["hour", "irr_bin"], observed=False)["power_mw"]
        .median()
        .reset_index()
        .rename(columns={"power_mw": "expected_floor_mw"})
    )
    floor_map: Dict[tuple[int, str], float] = {}
    for row in grouped.itertuples(index=False):
        key = (int(row.hour), str(row.irr_bin))
        val = float(row.expected_floor_mw)
        if np.isfinite(val) and val >= 0:
            floor_map[key] = min(val, ac_cap_mw)
    if not floor_map:
        return y_final

    out = np.asarray(y_final, dtype=float).copy()
    irr = pd.to_numeric(feat.get("Irradiation"), errors="coerce").to_numpy(dtype=float)
    hrs = pd.to_datetime(feat["ds"]).dt.hour.to_numpy(dtype=int)
    pred_bins = pd.cut(pd.Series(irr), bins=bin_edges, include_lowest=True, right=False)
    for i in range(len(out)):
        if not np.isfinite(irr[i]) or irr[i] < 750 or hrs[i] < 9 or hrs[i] > 16:
            continue
        floor_val = floor_map.get((int(hrs[i]), str(pred_bins.iloc[i])))
        if floor_val is None:
            continue
        out[i] = min(ac_cap_mw, max(out[i], float(floor_val)))
    return out


def _single_axis_tracker_profile_factor(feat: pd.DataFrame) -> np.ndarray:
    hours = pd.to_datetime(feat["ds"]).dt.hour.astype(int).to_numpy()
    irradiation = pd.to_numeric(feat["Irradiation"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    sun_elev = (
        pd.to_numeric(feat.get("sun_elev_deg", pd.Series(0.0, index=feat.index)), errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )

    daylight = (irradiation > 0.0) & (sun_elev > 0.0)
    factor = np.ones(len(feat), dtype=float)

    # Single-axis trackers usually broaden the daily profile: less pronounced
    # fixed-tilt noon peak, stronger shoulders for the same GHI.
    shoulder_mask = daylight & (
        ((hours >= 6) & (hours <= 10))
        | ((hours >= 15) & (hours <= 19))
    )
    edge_strength = np.clip((45.0 - sun_elev) / 45.0, 0.0, 1.0)
    factor[shoulder_mask] *= 1.0 + 0.16 * edge_strength[shoulder_mask]

    midday_mask = daylight & (hours >= 11) & (hours <= 14)
    high_sun_strength = np.clip((sun_elev - 35.0) / 35.0, 0.0, 1.0)
    factor[midday_mask] *= 1.0 - 0.08 * high_sun_strength[midday_mask]

    return np.clip(factor, 0.0, 1.18)


def _apply_single_axis_tracker_postprocessing(
    y_mw: np.ndarray,
    feat: pd.DataFrame,
    st: Station,
    capacity_mw: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Apply a conservative post-processing-only tracker correction.

    This intentionally does not change trained model inputs or the fixed-tilt
    path. The first rollout only reshapes the already-computed profile for
    single-axis tracker stations and clips by AC capacity plus historical p95/p99.
    """
    factor = _single_axis_tracker_profile_factor(feat)
    feat["tracker_profile_factor"] = factor
    feat["tracker_shoulder_boost"] = np.maximum(factor - 1.0, 0.0)
    feat["tracker_midday_flatten"] = np.maximum(1.0 - factor, 0.0)

    ac_cap_mw = _station_ac_nameplate_mw(st, capacity_mw)
    hist_cap_mw = _historical_tracker_output_cap_mw(st, ac_cap_mw)
    safe_cap_mw = float(
        np.clip(hist_cap_mw if hist_cap_mw is not None else ac_cap_mw, 0.0, ac_cap_mw)
    )

    base = np.asarray(y_mw, dtype=float)
    reshaped = base * factor
    hours = pd.to_datetime(feat["ds"]).dt.hour.astype(int).to_numpy()
    irradiation = pd.to_numeric(feat["Irradiation"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    sun_elev = (
        pd.to_numeric(feat.get("sun_elev_deg", pd.Series(0.0, index=feat.index)), errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    daylight = (irradiation > 0.0) & (sun_elev > 0.0)

    # A tracker station should not be represented only by the fixed-tilt output
    # multiplied by a tiny factor. Blend in a tracker-shaped reference built from
    # both irradiance and the station's own clear-day hourly median/p75/p95 shape.
    irradiance_reference = (
        ac_cap_mw * (irradiation / 1000.0) * _station_pr_default(st) * factor
    )
    hourly_profile = _historical_tracker_hourly_profile_mw(st, ac_cap_mw)
    historical_reference = np.full(len(feat), np.nan, dtype=float)
    for i, hour in enumerate(hours):
        stats = hourly_profile.get(int(hour))
        if not stats:
            continue
        historical_reference[i] = 0.35 * stats["median"] + 0.65 * stats["p75"]

    calibration_curve = _tracker_station_calibration_curve(st, ac_cap_mw)
    station_calibration = np.ones(len(feat), dtype=float)
    for i, hour in enumerate(hours):
        stats = calibration_curve.get(int(hour))
        if stats:
            station_calibration[i] = float(stats["factor"])

    # Station-calibrated expected power curve. This is the shared calibration
    # path for operational and postfact forecasts; only the weather input
    # should differ between those modes.
    calibrated_reference = np.clip(irradiance_reference * station_calibration, 0.0, ac_cap_mw)

    tracker_reference = calibrated_reference.copy()
    hist_mask = np.isfinite(historical_reference)
    tracker_reference[hist_mask] = (
        0.45 * calibrated_reference[hist_mask]
        + 0.55 * historical_reference[hist_mask]
    )

    plateau_mask = (
        daylight
        & (irradiation >= 500.0)
        & np.isin(hours, np.arange(8, 16))
        & hist_mask
    )
    tracker_reference[plateau_mask] = np.maximum(
        tracker_reference[plateau_mask],
        calibrated_reference[plateau_mask],
    )

    feat["tracker_reference_mw"] = tracker_reference
    feat["tracker_historical_reference_mw"] = historical_reference
    feat["tracker_station_calibration_factor"] = station_calibration
    feat["tracker_calibrated_expected_mw"] = calibrated_reference

    corrected = reshaped.copy()
    corrected[daylight] = 0.35 * reshaped[daylight] + 0.65 * tracker_reference[daylight]
    corrected = np.clip(corrected, 0.0, safe_cap_mw)
    return corrected, {
        "ac_cap_mw": float(ac_cap_mw),
        "safe_cap_mw": safe_cap_mw,
        "historical_profile_hours": float(len(hourly_profile)),
        "station_calibration_hours": float(len(calibration_curve)),
        "plateau_hours_applied": float(np.count_nonzero(plateau_mask)),
    }



def _tracker_station_calibration_curve(st: Station, ac_cap_mw: float) -> Dict[int, Dict[str, float]]:
    """Hourly station calibration factors derived from post-fact history.

    The same curve is intentionally used by operational and backfilled
    (postfact) runs, so tracker stations are not calibrated differently just
    because one run uses forecast weather and the other uses measured weather.
    Factors are based on actual AC power divided by a physical expected curve
    and clipped to a conservative range to avoid learning data glitches.
    """
    rows = list(
        SolarRecord.objects.filter(
            station=st,
            history_scope=SolarRecord.HISTORY_SCOPE_MAIN,
            power_kw__isnull=False,
            irradiation__isnull=False,
        )
        .order_by("-timestamp")
        .values("timestamp", "power_kw", "irradiation")[:24 * 365]
    )
    if len(rows) < 24:
        return {}

    hist = pd.DataFrame(rows)
    hist["timestamp"] = pd.to_datetime(hist["timestamp"], errors="coerce")
    hist["hour"] = hist["timestamp"].dt.hour.astype("Int64")
    hist["irr"] = pd.to_numeric(hist["irradiation"], errors="coerce")
    hist["power_mw"] = pd.to_numeric(hist["power_kw"], errors="coerce") / 1000.0
    hist = hist.dropna(subset=["hour", "irr", "power_mw"])
    hist = hist[(hist["irr"] >= 80.0) & (hist["power_mw"] >= 0.0)]
    if len(hist) < 24:
        return {}

    physical_expected = ac_cap_mw * (hist["irr"] / 1000.0) * _station_pr_default(st)
    physical_expected = physical_expected.clip(lower=max(ac_cap_mw * 0.03, 0.05), upper=ac_cap_mw)
    hist["calibration_factor"] = (hist["power_mw"] / physical_expected).replace([np.inf, -np.inf], np.nan)
    hist = hist.dropna(subset=["calibration_factor"])
    hist = hist[(hist["calibration_factor"] > 0.2) & (hist["calibration_factor"] < 2.0)]
    if len(hist) < 24:
        return {}

    curve: Dict[int, Dict[str, float]] = {}
    for hour, values in hist.groupby("hour")["calibration_factor"]:
        values = values.dropna()
        if len(values) < 3:
            continue
        factor = float(values.median())
        if not np.isfinite(factor):
            continue
        curve[int(hour)] = {
            "factor": float(np.clip(factor, 0.70, 1.30)),
            "samples": float(len(values)),
        }
    return curve


def _add_tracker_model_features(feat: pd.DataFrame, st: Station, capacity_mw: float) -> pd.DataFrame:
    """Add tracker-aware model regressors before XGB/NP prediction.

    Retrained tracker models can learn against the station-calibrated tracker
    expected curve instead of a fixed-tilt GHI curve. For non-tracker stations
    the columns are harmless defaults, so legacy models keep working.
    """
    out = feat.copy()
    out["tracker_profile_factor"] = 1.0
    out["tracker_station_calibration_factor"] = 1.0
    base_expected = pd.to_numeric(out.get("y_expected"), errors="coerce").fillna(0.0)
    out["tracker_calibrated_expected_mw"] = base_expected
    out["tracker_calibrated_expected_log"] = np.log1p(base_expected)

    if not _is_single_axis_tracker(st) or out.empty:
        return out

    try:
        from .tracker_pvlib_training import add_tracker_prediction_features

        pvlib_out = add_tracker_prediction_features(out, st, capacity_mw)
        pvlib_out["tracker_calibrated_expected_mw"] = pvlib_out["tracker_pvlib_baseline_mw"]
        pvlib_out["tracker_calibrated_expected_log"] = pvlib_out["tracker_pvlib_baseline_log"]
        return pvlib_out
    except Exception as exc:
        logger.exception(
            "[TRACKER_PVLIB] station %s failed to compute pvlib features; falling back to legacy tracker features: %s",
            st.pk,
            exc,
        )

    factor = _single_axis_tracker_profile_factor(out)
    ac_cap_mw = _station_ac_nameplate_mw(st, capacity_mw)
    irradiation = pd.to_numeric(out.get("Irradiation"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    hours = pd.to_datetime(out["ds"]).dt.hour.astype(int).to_numpy()

    calibration_curve = _tracker_station_calibration_curve(st, ac_cap_mw)
    station_calibration = np.ones(len(out), dtype=float)
    for i, hour in enumerate(hours):
        stats = calibration_curve.get(int(hour))
        if stats:
            station_calibration[i] = float(stats["factor"])

    calibrated_expected = np.clip(
        ac_cap_mw * (irradiation / 1000.0) * _station_pr_default(st) * factor * station_calibration,
        0.0,
        ac_cap_mw,
    )
    out["tracker_profile_factor"] = factor
    out["tracker_station_calibration_factor"] = station_calibration
    out["tracker_calibrated_expected_mw"] = calibrated_expected
    out["tracker_calibrated_expected_log"] = np.log1p(calibrated_expected)
    return out


def _describe_np_model(model: object) -> str:
    if model is None:
        return "model=None"
    parts = [
        f"type={type(model)}",
        f"has_predict={hasattr(model, 'predict')}",
        f"has_trainer={getattr(model, 'trainer', None) is not None}",
        f"has_model={getattr(model, 'model', None) is not None}",
        f"has_init_trainer={callable(getattr(model, '_init_trainer', None))}",
    ]
    return ", ".join(parts)


def _station_capacity_mw(st: Station) -> float:
    """
    Пытаемся достать мощность станции.
    Поддерживаем разные поля (потому что у тебя модели/миграции менялись).
    """
    capacity_ac_kw = None
    for name in ["capacity_ac_kw", "capacity_kw", "capacity_dc_kw"]:
        if hasattr(st, name) and getattr(st, name):
            capacity_ac_kw = float(getattr(st, name))
            break

    capacity_mw_from_fields = None
    for name in ["capacity_mw", "capacity_ac_mw"]:
        if hasattr(st, name) and getattr(st, name):
            capacity_mw = float(getattr(st, name))
            # Частая проблема: в поле capacity_mw попадает значение в kW (например 8800).
            # Если значение слишком большое, трактуем как kW и конвертируем в MW.
            if capacity_mw >= 1000:
                capacity_mw_from_fields = capacity_mw / 1000.0
                break
            if capacity_mw > 100 and capacity_ac_kw:
                capacity_mw_from_fields = capacity_mw / 1000.0
                break
            capacity_mw_from_fields = capacity_mw
            break

    if capacity_ac_kw:
        if capacity_mw_from_fields is None:
            capacity_mw_from_fields = capacity_ac_kw / 1000.0

    history_sources = [st]
    if getattr(st, "history_source_id", None):
        history_sources.append(st.history_source)

    hist_peak_kw = None
    for src in history_sources:
        qs = (
            SolarRecord.objects.filter(
                station=src,
                history_scope=SolarRecord.HISTORY_SCOPE_MAIN,
            )
            .exclude(power_kw__isnull=True)
            .order_by("-timestamp")
            .values_list("power_kw", flat=True)[:24 * 180]
        )
        if not qs:
            continue
        peak = float(np.nanmax(np.asarray(list(qs), dtype=float)))
        if np.isfinite(peak) and peak > 0:
            hist_peak_kw = max(hist_peak_kw or 0.0, peak)

    hist_peak_mw = (hist_peak_kw / 1000.0) if hist_peak_kw else None

    hist_uplift_factor_raw = getattr(settings, "FORECAST_HISTORY_CAPACITY_UPLIFT_FACTOR", 1.08)
    hist_uplift_guard_raw = getattr(settings, "FORECAST_HISTORY_CAPACITY_UPLIFT_GUARD", 1.8)
    try:
        hist_uplift_factor = float(hist_uplift_factor_raw)
    except (TypeError, ValueError):
        hist_uplift_factor = 1.08
    try:
        hist_uplift_guard = float(hist_uplift_guard_raw)
    except (TypeError, ValueError):
        hist_uplift_guard = 1.8

    if capacity_mw_from_fields and hist_peak_mw:
        # Защита от выбросов в истории (невалидные единицы/разовые аномалии):
        # не даём history-пику раздувать мощность в разы.
        if hist_peak_mw > capacity_mw_from_fields * hist_uplift_guard:
            logger.warning(
                "[FORECAST] station %s skip historical capacity uplift as outlier: field=%.3f MW, hist_peak=%.3f MW, guard=%.2fx",
                st.pk,
                capacity_mw_from_fields,
                hist_peak_mw,
                hist_uplift_guard,
            )
        elif hist_peak_mw > capacity_mw_from_fields * hist_uplift_factor:
            logger.warning(
                "[FORECAST] station %s capacity uplift from %.3f MW to historical peak %.3f MW",
                st.pk,
                capacity_mw_from_fields,
                hist_peak_mw,
            )
            return hist_peak_mw

    if capacity_mw_from_fields:
        return capacity_mw_from_fields

    if hist_peak_mw:
        return max(0.5, hist_peak_mw)

    # fallback: если нет поля — пусть будет 10MW, чтобы не было микроскопии
    return 10.0


def _solar_hours_from_history(st: Station) -> Tuple[int, int]:
    """
    Берём “солнечные часы” из истории:
    - ищем часы, где irradiation>50 или power_kw>0
    - берём min/max hour
    Всегда гарантируем широкий диапазон 5-20.
    """
    qs = SolarRecord.objects.filter(station=st, history_scope=SolarRecord.HISTORY_SCOPE_MAIN).order_by("-timestamp")[:14 * 24]
    if not qs.exists() and getattr(st, "history_source_id", None):
        qs = SolarRecord.objects.filter(station=st.history_source, history_scope=SolarRecord.HISTORY_SCOPE_MAIN).order_by("-timestamp")[:14 * 24]
    if not qs.exists():
        return (9, 17)

    df = pd.DataFrame.from_records(qs.values("timestamp", "irradiation", "power_kw"))
    if df.empty:
        return (9, 17)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    try:
        if getattr(df["timestamp"].dt, "tz", None) is not None:
            df["timestamp"] = df["timestamp"].dt.tz_convert(timezone.get_current_timezone())
    except Exception:
        pass
    df["hour"] = df["timestamp"].dt.hour
    mask = (df["irradiation"].fillna(0) > 50) | (df["power_kw"].fillna(0) > 0)
    if mask.sum() < 5:
        return (9, 17)

    hours = df.loc[mask, "hour"].astype(int)
    hmin = int(np.floor(hours.quantile(0.1)))
    hmax = int(np.ceil(hours.quantile(0.9)))
    # немного расширим; если окно узкое — берём фиксированный день 9-17
    h1 = max(5, hmin - 1)
    h2 = min(20, hmax + 1)
    if (h2 - h1) < 6:
        return (9, 17)
    return (h1, h2)


def _make_base_grid(days: int, solar_hours: Tuple[int, int]) -> pd.DataFrame:
    """
    Делает сетку часов на days вперёд (включая завтра), ограничивая "солнечными" часами.
    """
    now = timezone.localtime(timezone.now())
    try:
        h1, h2 = solar_hours
    except Exception:
        logger.warning("[FORECAST] invalid solar_hours=%s, fallback to (9, 17)", solar_hours)
        h1, h2 = 9, 17

    # начинаем с ближайшего следующего дня, чтобы не строить уже прошедшие часы
    start_date = (now + pd.Timedelta(days=1)).date()
    start = (
        timezone.datetime.combine(start_date, timezone.datetime.min.time())
        .replace(hour=0, tzinfo=now.tzinfo, minute=0, second=0, microsecond=0)
    )
    end = start + pd.Timedelta(days=days)

    all_hours = pd.date_range(start=start, end=end, freq="h", inclusive="left")
    df = pd.DataFrame({"ds": all_hours})
    df = df[(df["ds"].dt.hour >= h1) & (df["ds"].dt.hour <= h2)].copy()
    df["ds"] = df["ds"].dt.floor("h")
    return df.reset_index(drop=True)


def _solar_hours_from_weather(
    weather_df: pd.DataFrame,
    start_date: date,
    days: int,
) -> Optional[Tuple[int, int]]:
    if weather_df.empty or "irradiation" not in weather_df.columns:
        return None

    df = weather_df.copy()
    df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
    try:
        if getattr(df["ds"].dt, "tz", None) is not None:
            df["ds"] = df["ds"].dt.tz_convert(timezone.get_current_timezone())
    except Exception:
        pass

    end_date = start_date + pd.Timedelta(days=days)
    if hasattr(end_date, "date"):
        end_date = end_date.date()
    mask_date = (df["ds"].dt.date >= start_date) & (df["ds"].dt.date < end_date)
    df = df[mask_date]
    if df.empty:
        return None

    mask = df["irradiation"].fillna(0) > 50
    if mask.sum() < 3:
        return None

    hours = df.loc[mask, "ds"].dt.hour.astype(int)
    hmin = int(np.floor(hours.quantile(0.1)))
    hmax = int(np.ceil(hours.quantile(0.9)))
    h1 = max(5, hmin - 1)
    h2 = min(20, hmax + 1)

    winter_months = {11, 12, 1, 2}
    if start_date.month in winter_months:
        h1 = min(h1, 8)
        h2 = max(h2, 17)
        h1 = max(5, h1)
        h2 = min(20, h2)
    if (h2 - h1) < 6:
        return None
    return (h1, h2)




def _weather_from_history(st: Station, target_dates: set[date], forecast_scope: str = "main") -> pd.DataFrame:
    empty_df = pd.DataFrame(
        columns=["ds", "irradiation", "air_temp", "wind_speed", "cloudcover", "humidity", "precip", "snowfall", "snowdepth", "weather_code"]
    )
    if not target_dates:
        return empty_df

    preferred_scope = SolarRecord.HISTORY_SCOPE_MAIN if forecast_scope == "main" else SolarRecord.HISTORY_SCOPE_TEST
    scope_order = [preferred_scope]
    if preferred_scope != SolarRecord.HISTORY_SCOPE_MAIN:
        scope_order.append(SolarRecord.HISTORY_SCOPE_MAIN)

    station_order = [st]
    if getattr(st, "history_source_id", None):
        station_order.append(st.history_source)

    data = []
    for history_scope in scope_order:
        for source_station in station_order:
            qs = SolarRecord.objects.filter(station=source_station, history_scope=history_scope, timestamp__date__in=list(target_dates))
            data = list(qs.values("timestamp", "irradiation", "irradiation_ghi", "irradiation_poa", "air_temp"))
            if data:
                break
        if data:
            break

    if not data:
        return empty_df

    df = pd.DataFrame(data)
    df["ds"] = pd.to_datetime(df["timestamp"], errors="coerce").dt.floor("h")
    legacy_irr = pd.to_numeric(df.get("irradiation"), errors="coerce")
    ghi = pd.to_numeric(df.get("irradiation_ghi"), errors="coerce")
    poa = pd.to_numeric(df.get("irradiation_poa"), errors="coerce")
    source_irradiation_type = getattr(source_station, "irradiation_type", "GHI")
    if source_irradiation_type == "POA":
        df["irradiation"] = ghi
        df["irradiation_poa"] = poa.combine_first(legacy_irr)
    else:
        df["irradiation"] = ghi.combine_first(legacy_irr)
        df["irradiation_poa"] = poa
    df["air_temp"] = pd.to_numeric(df.get("air_temp"), errors="coerce")
    for c in ["wind_speed", "cloudcover", "humidity", "precip", "snowfall", "snowdepth", "weather_code"]:
        df[c] = np.nan

    return (
        df[["ds", "irradiation", "air_temp", "wind_speed", "cloudcover", "humidity", "precip", "snowfall", "snowdepth", "weather_code"]]
        .groupby("ds", as_index=False)
        .mean(numeric_only=True)
        .sort_values("ds")
        .reset_index(drop=True)
    )


def _make_base_grid_for_dates(target_dates: set[date], solar_hours: Tuple[int, int], tzinfo) -> pd.DataFrame:
    try:
        h1, h2 = solar_hours
    except Exception:
        h1, h2 = 9, 17

    rows = []
    for d in sorted(target_dates):
        for hour in range(int(h1), int(h2) + 1):
            rows.append(
                timezone.datetime.combine(d, timezone.datetime.min.time()).replace(
                    hour=hour, minute=0, second=0, microsecond=0, tzinfo=tzinfo
                )
            )

    return pd.DataFrame({"ds": pd.to_datetime(rows)}).reset_index(drop=True)

def _merge_weather(base: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    w = weather.copy()
    w["ds"] = pd.to_datetime(w["ds"]).dt.floor("h")
    base["ds"] = pd.to_datetime(base["ds"]).dt.floor("h")
    out = base.merge(w, on="ds", how="left")
    return out


def _merge_weather_with_hourly_profile_fallback(base: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    out = _merge_weather(base, weather)
    if weather.empty:
        return out

    if "irradiation" in out.columns and out["irradiation"].notna().any():
        return out

    w = weather.copy()
    w["ds"] = pd.to_datetime(w["ds"], errors="coerce")
    w = w.dropna(subset=["ds"])
    if w.empty:
        return out

    w["hour"] = w["ds"].dt.hour
    weather_cols = [
        "irradiation",
        "air_temp",
        "wind_speed",
        "cloudcover",
        "humidity",
        "precip",
        "snowfall",
        "snowdepth",
        "weather_code",
    ]
    for col in weather_cols:
        if col not in w.columns:
            w[col] = np.nan
    profile = w.groupby("hour", as_index=False)[weather_cols].mean(numeric_only=True)

    base_with_hour = base.copy()
    base_with_hour["ds"] = pd.to_datetime(base_with_hour["ds"], errors="coerce")
    base_with_hour["hour"] = base_with_hour["ds"].dt.hour
    projected = base_with_hour.merge(profile, on="hour", how="left")
    return projected.drop(columns=["hour"])


def _add_sun_geometry(df: pd.DataFrame, lat_deg: float) -> pd.DataFrame:
    lat = np.deg2rad(lat_deg)
    doy = df["ds"].dt.dayofyear
    hour = df["ds"].dt.hour
    hour_angle = np.deg2rad((hour - 12) * 15)
    decl = np.deg2rad(23.44) * np.sin(2 * np.pi * (284 + doy) / 365)
    sin_elev = (
        np.sin(lat) * np.sin(decl)
        + np.cos(lat) * np.cos(decl) * np.cos(hour_angle)
    )
    df["sun_elev_deg"] = np.rad2deg(np.arcsin(np.clip(sin_elev, -1, 1)))
    df["low_sun_flag"] = (df["sun_elev_deg"] < 15).astype(int)
    return df


def _compute_features(df: pd.DataFrame, capacity_mw: float, lat_deg: float) -> pd.DataFrame:
    """
    Генерим фичи под ожидаемый набор XGB/NP (v16 residual).
    """
    out = df.copy()

    # нормальные имена для XGB
    out["Irradiation"] = pd.to_numeric(out.get("irradiation"), errors="coerce").fillna(0.0)
    out["Air_Temp"] = pd.to_numeric(out.get("air_temp"), errors="coerce").fillna(0.0)
    out["Wind_Speed"] = pd.to_numeric(out.get("wind_speed"), errors="coerce")
    out["snowfall"] = pd.to_numeric(out.get("snowfall"), errors="coerce")
    out["snowdepth"] = pd.to_numeric(out.get("snowdepth"), errors="coerce")
    out["weather_code"] = pd.to_numeric(out.get("weather_code"), errors="coerce")

    # PV_Temp — если нет в погоде, аппроксимируем как в локальном скрипте
    out["PV_Temp"] = out["Air_Temp"] + np.maximum(out["Irradiation"] - 50, 0) / 1000 * 20

    out["hour"] = pd.to_datetime(out["ds"]).dt.hour.astype(int)
    out["month"] = pd.to_datetime(out["ds"]).dt.month.astype(int)

    noise_floor = _forecast_irradiation_noise_floor_wm2()
    out.loc[out["Irradiation"] < noise_floor, "Irradiation"] = 0.0

    # Мягкий прогнозный boost оставляем только для 06:00/09:00.
    # 07:00/08:00 больше не усиливаем: именно эти часы склонны к завышению.
    morning_boost = _forecast_morning_irradiation_boost()
    morning_mask = out["hour"].isin([6, 9]) & (out["Irradiation"] > 0)
    out.loc[morning_mask, "Irradiation"] = out.loc[morning_mask, "Irradiation"] * morning_boost

    out["sunrise_hour_flag"] = out["hour"].between(6, 8).astype(int)
    ramp_map = {6: 0.35, 7: 0.65, 8: 0.90}
    out["solar_ramp_factor"] = out["hour"].map(ramp_map).fillna(1.0).astype(float)
    out["irradiation_x_ramp"] = out["Irradiation"] * out["solar_ramp_factor"]

    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12.0)

    # простые флаги
    out["is_daylight"] = (out["Irradiation"] > 20).astype(int)

    out["is_clear"] = ((out["Irradiation"] > 200) & (out["Air_Temp"] > 0)).astype(int)

    out["morning_peak_boost"] = ((out["hour"] == 6) & (out["Irradiation"] > 80)).astype(int)
    out["evening_penalty"] = ((out["hour"] == 19) & (out["Irradiation"] > 39)).astype(int)
    out["overdrive_flag"] = ((out["Irradiation"] > 950) & (out["Air_Temp"] > 30)).astype(int)
    out["midday_penalty"] = ((out["hour"].isin([12, 13, 14]))).astype(int)
    out["is_morning_active"] = ((out["hour"] == 6) & (out["Irradiation"] > 49)).astype(int)

    # ожидаемая генерация и лог-таргет (как в обучении)
    expected_mw = (capacity_mw * (out["Irradiation"] / 1000.0) * PR_FOR_EXPECTED).clip(upper=capacity_mw * 0.95)
    out["y_expected"] = expected_mw
    out["y_expected_log"] = np.log1p(expected_mw * 0.95)

    out = _add_sun_geometry(out, lat_deg)

    winter_months = {11, 12, 1, 2}
    winter_mask = (out["month"].isin(winter_months)) & (out["sun_elev_deg"] > 0)
    out.loc[winter_mask, "Irradiation"] = out.loc[winter_mask, "Irradiation"].clip(lower=10)
    morning_winter_mask = winter_mask & out["hour"].isin([8, 9, 10])
    out.loc[morning_winter_mask, "Irradiation"] = out.loc[morning_winter_mask, "Irradiation"].clip(lower=20)

    # гарантируем порядок и наличие
    for c in XGB_EXPECTED_FEATURES:
        if c not in out.columns:
            out[c] = 0.0

    return out


def _compute_winter_factors(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    snowdepth = pd.to_numeric(out.get("snowdepth"), errors="coerce").fillna(0.0)
    snowfall = pd.to_numeric(out.get("snowfall"), errors="coerce").fillna(0.0)
    weather_code = pd.to_numeric(out.get("weather_code"), errors="coerce")
    month = pd.to_datetime(out.get("ds"), errors="coerce").dt.month

    auto_snow = (
        ((snowdepth >= AUTO_SNOWDEPTH_M_THRESHOLD) | (snowfall > 0) | (weather_code.isin(list(SNOW_CODES))))
        & (out["Air_Temp"] <= AUTO_TEMP_MAX_FOR_SNOW)
    )
    auto_fog = (weather_code.isin(list(FOG_CODES)) & (out["Air_Temp"] <= 0))

    out["auto_snow_flag"] = auto_snow.astype(int)
    out["auto_fog_flag"] = auto_fog.astype(int)

    factor = np.ones(len(out), dtype=float)
    factor[out["auto_fog_flag"] == 1] = np.minimum(factor[out["auto_fog_flag"] == 1], AUTO_FOG_FACTOR)
    factor[out["auto_snow_flag"] == 1] = np.minimum(factor[out["auto_snow_flag"] == 1], AUTO_SNOW_FACTOR)
    out["auto_winter_factor"] = factor
    return out


def _load_xgb_model(path: Path) -> Optional[Any]:
    xgb = _xgb_module()
    if xgb is None:
        return None
    try:
        booster = xgb.Booster()
        booster.load_model(str(path))
        return booster
    except Exception:
        return None


# =========================
# === NeuralProphet FIX ===
# =========================

def _allow_torch_safe_globals_for_np() -> None:
    """
    PyTorch 2.6: safe-unpickle режет pandas/neuralprophet объекты.
    Best-effort allowlist под разные версии pandas.
    """
    try:
        from torch.serialization import add_safe_globals

        from neuralprophet.forecaster import NeuralProphet
        from neuralprophet.configure import Normalization
        from neuralprophet.df_utils import ShiftScale

        from pandas._libs.tslibs import timestamps as _ts
        from pandas._libs.tslibs import timedeltas as _td

        allow = [NeuralProphet, Normalization, ShiftScale]

        # timestamp helper variants
        for name in ("_unpickle_timestamp", "_timestamp_unpickle"):
            fn = getattr(_ts, name, None)
            if fn is not None:
                allow.append(fn)

        # timedelta helper variants (в твоих ошибках часто именно _timedelta_unpickle)
        for name in ("_unpickle_timedelta", "_timedelta_unpickle"):
            fn = getattr(_td, name, None)
            if fn is not None:
                allow.append(fn)

        add_safe_globals(allow)
    except Exception:
        return


def _load_np_model(path: Path):
    """
    Грузим .np через neuralprophet.load() (правильный формат — model.save()).
    Fallback: torch.load(weights_only=False).

    ВАЖНО: если загружается объект NeuralProphet, но obj.model == None,
    то predict() падает 'NoneType has no attribute predict'. Ловим это здесь и даём ясную причину.
    """
    _allow_torch_safe_globals_for_np()
    torch_err: Optional[str] = None
    np_err: Optional[str] = None

    def _extract(m: object) -> Optional[object]:
        if m is None:
            return None
        if hasattr(m, "predict"):
            return m
        if isinstance(m, (tuple, list)):
            for itm in m:
                cand = _extract(itm)
                if cand is not None:
                    return cand
            return None
        if isinstance(m, dict):
            for key in ("model", "forecaster", "np_model", "forecast_model"):
                cand = _extract(m.get(key))
                if cand is not None:
                    return cand
            for v in m.values():
                cand = _extract(v)
                if cand is not None:
                    return cand
            return None
        return None

    def _validate_np(obj: object, src: str) -> object:
        if obj is None or not hasattr(obj, "predict"):
            raise TypeError(f"NP load returned invalid object from {src}: {type(obj)}")

        # КЛЮЧЕВО: внутренний torch-модуль должен быть восстановлен
        inner = getattr(obj, "model", None)
        if inner is None:
            raise TypeError(
                "NeuralProphet loaded but internal `model` is None (weights not restored). "
                "Обычно это значит: файл НЕ настоящий .np от `model.save()`, "
                "или повреждён/не той версии. Пересохрани модель через `model.save('...np')`."
            )
        return obj

    # 1) native NP loader
    try:
        loaded = np_load(str(path))
        model = _extract(loaded)
        if model is not None:
            return _validate_np(model, f"neuralprophet.load({path.name})")
    except Exception as e:
        np_err = str(e)

    # 2) torch fallback
    try:
        import torch
        loaded = torch.load(str(path), map_location="cpu", weights_only=False)
        model = _extract(loaded)
        if model is not None:
            return _validate_np(model, f"torch.load({path.name})")
    except Exception as e:
        torch_err = str(e)

    raise TypeError(f"NP load failed: np_err={np_err}, torch_err={torch_err}")


def _predict_np(
    model,
    df_feat: pd.DataFrame,
    reg_features: Optional[List[str]] = None,
    cap_for_expected: Optional[float] = None,
    fill_map: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Предикт NeuralProphet:
    - model.predict ожидает df с 'ds' и будущими регрессорами, если они были при обучении.
    Тут мы подаём минимум: ds + регрессоры Irradiation/Air_Temp/PV_Temp и т.п.
    """
    if model is None or not hasattr(model, "predict"):
        raise TypeError("NP model is not loaded or has no predict() method")

    if getattr(model, "trainer", None) is None:
        init_trainer = getattr(model, "_init_trainer", None)
        restore_trainer = getattr(model, "restore_trainer", None)
        errors: List[str] = []
        if callable(restore_trainer):
            try:
                trainer_obj = restore_trainer()
                if trainer_obj is not None and getattr(model, "trainer", None) is None:
                    model.trainer = trainer_obj
            except Exception as exc:
                errors.append(f"restore_trainer: {exc}")
        if getattr(model, "trainer", None) is None and callable(init_trainer):
            try:
                trainer_obj = init_trainer()
                if trainer_obj is not None and getattr(model, "trainer", None) is None:
                    model.trainer = trainer_obj
            except TypeError as exc:
                errors.append(f"default: {exc}")
                try:
                    trainer_obj = init_trainer(max_epochs=1)
                    if trainer_obj is not None and getattr(model, "trainer", None) is None:
                        model.trainer = trainer_obj
                except Exception as exc2:
                    errors.append(f"max_epochs=1: {exc2}")
            except Exception as exc:
                errors.append(f"default: {exc}")
        if getattr(model, "trainer", None) is None:
            details = f" Ошибка инициализации: {', '.join(errors)}" if errors else ""
            logger.warning(
                "[NP] NeuralProphet loaded without trainer (predict cannot run). "
                "Пересохрани модель через `model.save('...np')` или переобучи.%s | %s",
                details,
                _describe_np_model(model),
            )
            return np.full(len(df_feat), np.nan)

    df_feat = df_feat.copy()

    reg_list = reg_features or [
        "Irradiation",
        "Air_Temp",
        "PV_Temp",
        "hour_sin",
        "month_sin",
        "is_daylight",
        "is_clear",
        "morning_peak_boost",
        "overdrive_flag",
        "midday_penalty",
        "y_expected_log",
        "sunrise_hour_flag",
        "solar_ramp_factor",
        "irradiation_x_ramp",
    ]

    allowed_regressors = None
    if hasattr(model, "config_regressors"):
        config_regs = getattr(model, "config_regressors", None)
        regs = getattr(config_regs, "regressors", None) if config_regs is not None else None
        if isinstance(regs, dict):
            allowed_regressors = list(regs.keys())

    if allowed_regressors is not None:
        unexpected = [c for c in reg_list if c not in allowed_regressors]
        if unexpected:
            logger.warning("[NP] regressors not in model config -> dropped: %s", unexpected)
        reg_list = allowed_regressors

    # если нет y_expected_log, посчитаем на основе irradiation и мощности
    if "y_expected_log" not in df_feat.columns and "Irradiation" in df_feat.columns:
        cap_use = float(cap_for_expected) if cap_for_expected is not None else 1.0
        expected_mw = (cap_use * (df_feat["Irradiation"] / 1000.0) * PR_FOR_EXPECTED).clip(0, cap_use * 0.95)
        df_feat["y_expected_log"] = np.log1p(expected_mw)

    dfp = pd.DataFrame({"ds": pd.to_datetime(df_feat["ds"])})
    # y нужен для некоторых версий NP даже в будущем — кладём NaN
    dfp["y"] = np.nan

    missing = []
    for col in reg_list:
        if col in df_feat.columns:
            series = pd.to_numeric(df_feat[col], errors="coerce")
            dfp[col] = series.fillna(float((fill_map or {}).get(col, 0.0))).values
        else:
            missing.append(col)
            dfp[col] = float((fill_map or {}).get(col, 0.0))

    if missing:
        logger.warning("[NP] missing regressors -> filled with defaults: %s", missing)

    try:
        fcst = model.predict(dfp)
    except Exception as exc:
        logger.exception("[NP] predict failed; skipping NP for this forecast run: %s", exc)
        return np.full(len(dfp), np.nan)

    # NeuralProphet usually returns yhat1. Some versions can also return a
    # different row count for irregular hourly grids. Always align back to the
    # requested dfp["ds"] so a bad/odd NP output cannot crash saving rows.
    yhat_col = "yhat1" if "yhat1" in fcst.columns else None
    if not yhat_col:
        yhat_cols = [c for c in fcst.columns if c.startswith("yhat")]
        yhat_col = yhat_cols[0] if yhat_cols else None

    if not yhat_col:
        logger.warning("[NP] predict returned no yhat columns: %s", list(fcst.columns))
        return np.full(len(dfp), np.nan)

    yhat = pd.to_numeric(fcst[yhat_col], errors="coerce")
    if len(yhat) == len(dfp):
        return yhat.to_numpy(dtype=float)

    if "ds" in fcst.columns:
        aligned = (
            pd.DataFrame({"ds": pd.to_datetime(dfp["ds"]).dt.floor("h")})
            .merge(
                pd.DataFrame(
                    {
                        "ds": pd.to_datetime(fcst["ds"], errors="coerce").dt.floor("h"),
                        "yhat": yhat,
                    }
                ).dropna(subset=["ds"]).drop_duplicates(subset=["ds"], keep="last"),
                on="ds",
                how="left",
            )["yhat"]
        )
        logger.warning(
            "[NP] predict row-count mismatch input=%s output=%s; aligned by ds",
            len(dfp),
            len(fcst),
        )
        return pd.to_numeric(aligned, errors="coerce").to_numpy(dtype=float)

    logger.warning(
        "[NP] predict row-count mismatch input=%s output=%s and no ds column; padding/truncating",
        len(dfp),
        len(fcst),
    )
    out = np.full(len(dfp), np.nan, dtype=float)
    values = yhat.to_numpy(dtype=float)
    out[: min(len(out), len(values))] = values[: min(len(out), len(values))]
    return out


def _predict_xgb(booster: Any, df_feat: pd.DataFrame, feature_names: List[str]) -> np.ndarray:
    xgb = _xgb_module()
    if xgb is None:
        return np.zeros(len(df_feat), dtype=float)
    X = df_feat[feature_names].astype(float)
    dmat = xgb.DMatrix(X, feature_names=feature_names)
    pred = booster.predict(dmat)
    return pred


def _postprocess_xgb_prediction(
    pred: np.ndarray,
    xgb_meta: Dict,
    capacity_mw: float,
    df_feat: Optional[pd.DataFrame] = None,
) -> np.ndarray:
    """
    Приводим output XGB к MW.

    Поддерживаем 2 современных target:
    1) y_permw = y / cap_mw                -> pred * cap_mw_used
    2) y_over_expected = y / max(expected) -> pred * max(y_expected, floor)

    Для legacy-моделей в MW оставляем как есть.
    """
    out = np.asarray(pred, dtype=float)
    target = str((xgb_meta or {}).get("target", "")).lower()

    calib_mult = (xgb_meta or {}).get("xgb_calib_mult", 1.0)
    try:
        calib_mult = float(calib_mult)
    except (TypeError, ValueError):
        calib_mult = 1.0
    calib_mult = float(np.clip(calib_mult, 0.25, 12.0))

    is_over_expected = ("over_expected" in target) or ("y / max(y_expected" in target)
    if is_over_expected:
        cap_used = (xgb_meta or {}).get("cap_mw_used")
        try:
            cap_used = float(cap_used)
        except (TypeError, ValueError):
            cap_used = float(capacity_mw)
        if cap_used <= 0:
            cap_used = float(capacity_mw) if capacity_mw > 0 else 1.0

        floor = (xgb_meta or {}).get("y_expected_floor_mw")
        try:
            floor = float(floor)
        except (TypeError, ValueError):
            floor = max(0.05 * cap_used, 0.15)
        floor = max(floor, 1e-3)

        expected_col = str((xgb_meta or {}).get("expected_col", "") or "")
        if df_feat is None:
            expected = np.full(len(out), floor, dtype=float)
        elif expected_col and expected_col in df_feat.columns:
            expected = pd.to_numeric(df_feat.get(expected_col), errors="coerce").fillna(0.0).to_numpy(dtype=float)
            expected = np.maximum(expected, floor)
        else:
            irr = pd.to_numeric(df_feat.get("Irradiation"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
            expected = np.clip(cap_used * (irr / 1000.0) * PR_FOR_EXPECTED, 0.0, cap_used * 0.95)
            expected = np.maximum(expected, floor)
        return out * expected * calib_mult

    is_per_mw = (
        "per_mw" in target
        or "permw" in target
        or "y / cap_mw" in target
    )
    if not is_per_mw:
        return out

    cap_used = (xgb_meta or {}).get("cap_mw_used")
    try:
        cap_used = float(cap_used)
    except (TypeError, ValueError):
        cap_used = float(capacity_mw)
    if cap_used <= 0:
        cap_used = float(capacity_mw) if capacity_mw > 0 else 1.0

    return out * cap_used * calib_mult




def _xgb_is_systematically_low(
    y_xgb: np.ndarray,
    y_heur: np.ndarray,
    y_np: np.ndarray,
    np_ok: bool,
    capacity_mw: float,
) -> bool:
    """
    Детектор аномально низкого XGB относительно других сигналов.

    Срабатывает только когда NP доступен (иначе XGB может быть единственной моделью).
    Если в большинстве "дневных" точек XGB существенно ниже reference,
    считаем XGB недостоверным для данного прогона.
    """
    if not np_ok:
        return False

    x = np.asarray(y_xgb, dtype=float)
    h = np.asarray(y_heur, dtype=float)
    n = np.asarray(y_np, dtype=float)
    if len(x) == 0:
        return False

    ref = np.maximum(h, n)
    # интересуют точки, где есть значимая генерация
    daytime = ref >= max(0.15 * float(capacity_mw), 0.2)
    if daytime.sum() < 3:
        return False

    ratio = np.divide(x, np.maximum(ref, 1e-6))
    low = daytime & (ratio < 0.35)
    return float(low.mean()) >= 0.5

def _heuristic_mw(df_feat: pd.DataFrame, capacity_mw: float) -> np.ndarray:
    """
    Простая эвристика: мощность ~ irradiation/1000 * capacity * k
    k подбираем грубо, чтобы не было микроскопии.
    """
    irr = df_feat["Irradiation"].astype(float).to_numpy()
    p = (irr / 1000.0) * capacity_mw
    p = np.clip(p, 0, capacity_mw)
    return p




def _target_offsets_for_weekday_calendar(now_dt) -> List[int]:
    weekday = now_dt.weekday()
    if weekday == 4:  # Friday
        # По требованию календарного режима:
        # Пятница -> Вс + Пн + Вт
        return [2, 3, 4]
    if weekday in {0, 1, 2, 3}:  # Mon-Thu
        return [2]
    return []


def _historical_early_morning_caps_mw(st: Station, hours: Tuple[int, ...] = (7, 8)) -> Dict[int, float]:
    """
    Возвращает мягкие верхние ограничения для раннего утра по истории станции.

    Кап строится только для 07:00/08:00, по 95-му перцентилю фактической
    мощности с небольшим uplift. Если истории по часу мало, ограничение не
    применяется.
    """
    min_samples = _forecast_early_morning_cap_min_samples()
    uplift = _forecast_early_morning_cap_uplift()
    qs = SolarRecord.objects.filter(
        station=st,
        history_scope=SolarRecord.HISTORY_SCOPE_MAIN,
        timestamp__hour__in=list(hours),
        power_kw__isnull=False,
    ).values("timestamp", "power_kw")
    rows = list(qs)
    if not rows:
        return {}

    hist = pd.DataFrame(rows)
    hist["hour"] = pd.to_datetime(hist["timestamp"]).dt.hour.astype(int)
    hist["power_mw"] = (pd.to_numeric(hist["power_kw"], errors="coerce") / 1000.0).clip(lower=0.0)

    caps: Dict[int, float] = {}
    for hour in hours:
        values = hist.loc[hist["hour"] == hour, "power_mw"].dropna()
        if len(values) < min_samples:
            continue
        cap = float(values.quantile(0.95) * uplift)
        if np.isfinite(cap) and cap > 0:
            caps[int(hour)] = cap
    return caps


def _apply_early_morning_history_cap(
    y_mw: np.ndarray,
    feat: pd.DataFrame,
    st: Station,
    capacity_mw: float,
) -> Tuple[np.ndarray, Dict[int, float]]:
    caps = _historical_early_morning_caps_mw(st)
    if not caps:
        return np.asarray(y_mw, dtype=float), caps

    out = np.asarray(y_mw, dtype=float).copy()
    hours = pd.to_datetime(feat["ds"]).dt.hour.astype(int).to_numpy()
    for hour, cap in caps.items():
        safe_cap = float(np.clip(cap, 0.0, capacity_mw))
        if safe_cap <= 0:
            continue
        mask = hours == int(hour)
        out[mask] = np.minimum(out[mask], safe_cap)
    return out, caps


def run_forecast_for_station(
    station_id: int,
    days: int = 7,
    providers: Optional[List[str]] = None,
    manual_snow_enable: bool = False,
    manual_snow_factor: Optional[float] = None,
    manual_snow_dates: Optional[List[date]] = None,
    use_models: bool = True,
    horizon_mode: str = "weekday_calendar",
    forecast_scope: str = "main",
    target_dates: Optional[List[date]] = None,
) -> Dict:
    st = Station.objects.get(pk=station_id)
    capacity_mw = _station_capacity_mw(st)
    now = timezone.localtime(timezone.now())
    data_shift_hours = _station_data_shift_hours(st)
    forecast_shift_hours = _station_forecast_shift_hours(st)

    # Keep tracker diagnostics defined for every execution path. This prevents
    # NameError in the final result if conflict resolution or a non-tracker path
    # skips the tracker branch below.
    tracker_caps: Dict[str, float] = {}
    tracker_postprocessing_applied = False

    requested_target_dates = {d for d in (target_dates or []) if isinstance(d, date)}
    target_dates: Optional[set[date]] = requested_target_dates or None
    effective_days = max(int(days or 1), 1)
    if target_dates:
        min_date = min(target_dates)
        max_date = max(target_dates)
        effective_days = max((max_date - min_date).days + 1, 1)
    elif horizon_mode == "weekday_calendar":
        offsets = _target_offsets_for_weekday_calendar(now)
        if offsets:
            target_dates = {(now + pd.Timedelta(days=offset)).date() for offset in offsets}
            effective_days = max(offsets)

    # ---- погода ----
    weather_source = "fallback_zero"
    weather_df = pd.DataFrame(columns=["ds", "irradiation", "air_temp", "wind_speed", "cloudcover", "humidity", "precip"])

    lat = getattr(st, "lat", None) or getattr(st, "latitude", None)
    lon = getattr(st, "lon", None) or getattr(st, "longitude", None)
    tz_name = getattr(st, "timezone", None) or str(timezone.get_current_timezone())

    if lat is not None and lon is not None:
        provider_list = providers or getattr(settings, "FORECAST_WEATHER_PROVIDERS", ["visual_crossing"])
        fetchers = {
            "visual_crossing": fetch_visual_crossing_hourly,
            "open_meteo": fetch_open_meteo_hourly,
        }
        for provider in provider_list:
            fetcher = fetchers.get(provider)
            if fetcher is None:
                logger.warning("[FORECAST] unknown weather provider: %s", provider)
                continue
            if provider == "open_meteo":
                wres = fetcher(float(lat), float(lon), days=effective_days, tz_name=tz_name)
            else:
                wres = fetcher(float(lat), float(lon), days=effective_days)
            if wres.ok and not wres.df.empty:
                weather_source = wres.source
                weather_df = wres.df.copy()
                break

    start_date = min(target_dates) if target_dates else (now + pd.Timedelta(days=1)).date()

    if target_dates and max(target_dates) < (now + pd.Timedelta(days=1)).date():
        history_weather_df = _weather_from_history(st, target_dates, forecast_scope=forecast_scope)
        if not history_weather_df.empty:
            weather_df = history_weather_df
            weather_source = "history_backfill"

    solar_hours = _solar_hours_from_weather(weather_df, start_date, effective_days) or _solar_hours_from_history(st)

    if target_dates:
        base = _make_base_grid_for_dates(target_dates, solar_hours=solar_hours, tzinfo=now.tzinfo)
    else:
        base = _make_base_grid(days=effective_days, solar_hours=solar_hours)
    merged = _merge_weather_with_hourly_profile_fallback(base, weather_df)
    lat_deg = float(lat) if lat is not None else 47.86
    feat = _compute_features(merged, capacity_mw, lat_deg)
    feat = _compute_winter_factors(feat)

    if target_dates:
        ds_dates = pd.to_datetime(feat["ds"]).dt.date
        feat = feat.loc[ds_dates.isin(target_dates)].copy()

    report_days = len(target_dates) if target_dates else effective_days

    # После фильтрации по датам индекс может быть разреженным (например, начинаться с 11),
    # а массивы предсказаний индексируются позиционно с 0. Выравниваем индекс,
    # чтобы избежать ошибок вида "index X is out of bounds for axis 0" при сохранении.
    feat = feat.reset_index(drop=True)
    feat = _add_tracker_model_features(feat, st, capacity_mw)

    if feat.empty:
        return {
            "ok": True,
            "count": 0,
            "days": report_days,
            "solar_hours": list(solar_hours),
            "weather_source": weather_source,
            "np_ok": False,
            "xgb_ok": False,
            "np_error": "NO_TARGET_DATES",
            "xgb_error": "NO_TARGET_DATES",
            "horizon_mode": horizon_mode,
            "target_dates": sorted(str(d) for d in (target_dates or [])),
        }

    # ---- load models ----
    paths = _model_paths_for_station(st)
    np_path = paths["np"] if paths["np"].exists() else paths["legacy_np"]
    xgb_path = paths["xgb"] if paths["xgb"].exists() else paths["legacy_xgb"]
    np_meta_path = (
        paths["np_meta"] if paths["np_meta"].exists() else paths["legacy_np_meta"]
    )
    xgb_meta_path = (
        paths["xgb_meta"] if paths["xgb_meta"].exists() else paths["legacy_xgb_meta"]
    )

    np_meta: Dict = {}
    if np_meta_path.exists():
        try:
            np_meta = json.loads(np_meta_path.read_text(encoding="utf-8"))
        except Exception:
            np_meta = {}

    xgb_meta: Dict = {}
    if xgb_meta_path.exists():
        try:
            xgb_meta = json.loads(xgb_meta_path.read_text(encoding="utf-8"))
        except Exception:
            xgb_meta = {}

    stale_days_raw = getattr(settings, "FORECAST_AUTORETRAIN_STALE_DAYS", 14)
    try:
        stale_days = int(stale_days_raw)
    except (TypeError, ValueError):
        stale_days = 14
    np_stale = _model_file_is_stale(np_path, stale_days)
    xgb_stale = _model_file_is_stale(xgb_path, stale_days)

    if use_models and (not np_path.exists() or not xgb_path.exists() or np_stale or xgb_stale):
        try:
            from .train_models import train_models_for_station

            logger.info(
                "[MODEL] auto-train triggered (np_exists=%s, xgb_exists=%s, np_stale=%s, xgb_stale=%s, stale_days=%s).",
                np_path.exists(),
                xgb_path.exists(),
                np_stale,
                xgb_stale,
                stale_days,
            )
            _, np_path_new, xgb_path_new = train_models_for_station(st)
            if np_path_new is not None:
                np_path = np_path_new
                np_meta_path = np_path.with_suffix(".meta.json")
            if xgb_path_new is not None:
                xgb_path = xgb_path_new
                xgb_meta_path = xgb_path.with_suffix(".meta.json")
        except Exception as exc:
            logger.exception("[MODEL] auto-train failed: %s", exc)
        else:
            if np_meta_path.exists():
                try:
                    np_meta = json.loads(np_meta_path.read_text(encoding="utf-8"))
                except Exception:
                    np_meta = {}
            if xgb_meta_path.exists():
                try:
                    xgb_meta = json.loads(xgb_meta_path.read_text(encoding="utf-8"))
                except Exception:
                    xgb_meta = {}

    tracker_pvlib_pipeline = _is_single_axis_tracker(st) and (
        str((xgb_meta or {}).get("pipeline", "")) == "tracker_pvlib_v1"
        or str((np_meta or {}).get("pipeline", "")) == "tracker_pvlib_v1"
    )
    if tracker_pvlib_pipeline:
        try:
            from .tracker_pvlib_training import add_tracker_prediction_features

            tracker_meta = xgb_meta if xgb_meta.get("baseline_curve") else np_meta
            feat = add_tracker_prediction_features(feat, st, capacity_mw, tracker_meta)
            feat["tracker_calibrated_expected_mw"] = feat["tracker_pvlib_baseline_mw"]
            feat["tracker_calibrated_expected_log"] = feat["tracker_pvlib_baseline_log"]
        except Exception as exc:
            tracker_pvlib_pipeline = False
            logger.exception("[TRACKER_PVLIB] station %s failed to apply trained pvlib meta: %s", st.pk, exc)

    fallback_station = Station.objects.filter(pk=1).first()
    if fallback_station:
        fallback_paths = _model_paths_for_station(fallback_station)
        fallback_np_path = (
            fallback_paths["np"]
            if fallback_paths["np"].exists()
            else fallback_paths["legacy_np"]
        )
        fallback_np_meta_path = (
            fallback_paths["np_meta"]
            if fallback_paths["np_meta"].exists()
            else fallback_paths["legacy_np_meta"]
        )
        fallback_xgb_path = (
            fallback_paths["xgb"]
            if fallback_paths["xgb"].exists()
            else fallback_paths["legacy_xgb"]
        )
        fallback_xgb_meta_path = (
            fallback_paths["xgb_meta"]
            if fallback_paths["xgb_meta"].exists()
            else fallback_paths["legacy_xgb_meta"]
        )
    else:
        fallback_np_path = MODEL_DIR / "np_model_1.np"
        fallback_np_meta_path = MODEL_DIR / "np_model_1.meta.json"
        fallback_xgb_path = MODEL_DIR / "xgb_model_1.json"
        fallback_xgb_meta_path = MODEL_DIR / "xgb_model_1.meta.json"

    np_ok = False
    xgb_ok = False
    np_error = None
    xgb_error = None

    y_np = np.full(len(feat), np.nan)
    y_xgb = np.full(len(feat), np.nan)

    if use_models:
        # XGB
        booster = None
        if xgb_path.exists():
            booster = _load_xgb_model(xgb_path)
            if booster is None:
                xgb_error = f"XGB load failed: {xgb_path}"
                logger.warning("[XGB] load failed from %s", xgb_path)
        elif abs(capacity_mw - 8.8) < 0.05 and fallback_xgb_path.exists():
            booster = _load_xgb_model(fallback_xgb_path)
            if booster is None:
                xgb_error = f"XGB load failed: {fallback_xgb_path}"
                logger.warning("[XGB] load failed from %s", fallback_xgb_path)
            if fallback_xgb_meta_path.exists():
                try:
                    xgb_meta = json.loads(fallback_xgb_meta_path.read_text(encoding="utf-8"))
                except Exception:
                    xgb_meta = xgb_meta
        else:
            xgb_error = f"XGB model not found: {xgb_path}"
            logger.warning("[XGB] model not found: %s", xgb_path)

        if booster is not None:
            try:
                feature_names = xgb_meta.get("X_cols") or XGB_EXPECTED_FEATURES
                y_xgb = _predict_xgb(booster, feat, feature_names)
                y_xgb = _postprocess_xgb_prediction(y_xgb, xgb_meta, capacity_mw=capacity_mw, df_feat=feat)
                xgb_ok = True
            except Exception as e:
                xgb_error = str(e)
                xgb_ok = False
                booster = None

        # NP (FIXED)
        if np_path.exists():
            try:
                model = _load_np_model(np_path)
                logger.info("[NP] loaded from %s %s", np_path, _describe_np_model(model))
                y_np = _predict_np(
                    model,
                    feat,
                    reg_features=np_meta.get("features_reg"),
                    cap_for_expected=np_meta.get("cap_mw") or np_meta.get("cap_mw_used"),
                    fill_map=np_meta.get("fill_map") if isinstance(np_meta.get("fill_map"), dict) else None,
                )
                np_ok = True
            except Exception as e:
                logger.exception("[NP] ERROR: %s", e)
                np_error = str(e)
                np_ok = False
        elif abs(capacity_mw - 8.8) < 0.05 and fallback_np_path.exists():
            try:
                model = _load_np_model(fallback_np_path)
                if fallback_np_meta_path.exists():
                    try:
                        np_meta = json.loads(fallback_np_meta_path.read_text(encoding="utf-8"))
                    except Exception:
                        np_meta = np_meta
                logger.info("[NP] loaded from %s %s", fallback_np_path, _describe_np_model(model))
                y_np = _predict_np(
                    model,
                    feat,
                    reg_features=np_meta.get("features_reg"),
                    cap_for_expected=np_meta.get("cap_mw") or np_meta.get("cap_mw_used"),
                    fill_map=np_meta.get("fill_map") if isinstance(np_meta.get("fill_map"), dict) else None,
                )
                np_ok = True
            except Exception as e:
                logger.exception("[NP] ERROR: %s", e)
                np_error = str(e)
                np_ok = False
        else:
            np_error = f"NP model not found: {np_path}"
            logger.warning("[NP] model not found: %s", np_path)
    else:
        np_error = "NP skipped: Open-Meteo only"
        xgb_error = "XGB skipped: Open-Meteo only"

    # эвристика (MW)
    y_heur = feat.get("y_expected")
    if y_heur is None:
        y_heur = _heuristic_mw(feat, capacity_mw=capacity_mw)
    else:
        y_heur = y_heur.to_numpy(dtype=float)

    # NP теперь выдаёт residual -> приводим к полной мощности.
    # Tracker models retrained after this change can store residuals relative to
    # tracker_calibrated_expected_mw, so operational and postfact use the same
    # tracker base curve.
    if np_ok:
        np_base_col = str((np_meta or {}).get("base_expected_col", "y_expected") or "y_expected")
        np_base = feat.get(np_base_col) if np_base_col in feat.columns else feat.get("y_expected", 0.0)
        y_np = y_np + np.nan_to_num(np_base)

    # чистим NaN до ансамбля, иначе NaN в XGB/NP зануляет итог
    if use_models:
        y_np = np.nan_to_num(y_np, nan=0.0)
        y_xgb = np.nan_to_num(y_xgb, nan=0.0)
    y_heur = np.nan_to_num(y_heur, nan=0.0)

    # КЛЮЧЕВО: ограничиваем модельные предсказания ДО ансамбля.
    # Иначе отрицательный NP/XGB может занулить ранние утренние часы в y_final.
    if use_models:
        y_np = np.clip(y_np, 0, capacity_mw)
        y_xgb = np.clip(y_xgb, 0, capacity_mw)

    # если XGB системно сильно ниже NP/эвристики, не даём ему тянуть итог вниз
    if use_models and xgb_ok and _xgb_is_systematically_low(y_xgb, y_heur, y_np, np_ok=np_ok, capacity_mw=capacity_mw):
        logger.warning("[XGB] low-confidence: systematically below NP/heuristic, skip in ensemble")
        xgb_ok = False
        xgb_error = "XGB low-confidence vs NP/heuristic"

    # ансамбль:
    y_final = y_heur.copy()
    if use_models:
        if xgb_ok:
            y_final = 0.6 * y_heur + 0.4 * y_xgb
        if np_ok and xgb_ok:
            y_final = 0.2 * y_heur + 0.4 * y_xgb + 0.4 * y_np
        elif np_ok and not xgb_ok:
            y_final = 0.6 * y_heur + 0.4 * y_np

    # клип по мощности станции (MW) и перевод в кВт для сохранения
    y_heur = np.clip(y_heur, 0, capacity_mw)
    y_final = np.clip(np.nan_to_num(y_final, nan=0.0), 0, capacity_mw)
    y_final = np.minimum(
        y_final,
        np.maximum.reduce(
            [
                y_heur,
                y_xgb,
                (capacity_mw * (feat["Irradiation"].to_numpy() / 1000.0) * PR_FOR_EXPECTED * 1.25),
            ]
        ),
    )

    clear_sky_floor_ratio = _forecast_clear_sky_floor_ratio()
    cloudcover_raw = feat["cloudcover"] if "cloudcover" in feat.columns else pd.Series(np.nan, index=feat.index)
    cloudcover_series = pd.to_numeric(cloudcover_raw, errors="coerce")
    clear_mask = (
        (feat["Irradiation"].to_numpy(dtype=float) >= 280.0)
        & (cloudcover_series.fillna(100.0).to_numpy(dtype=float) <= 35.0)
    )
    clear_floor = np.clip(y_heur * clear_sky_floor_ratio, 0, capacity_mw)
    y_final = np.where(clear_mask, np.maximum(y_final, clear_floor), y_final)

    y_final = np.clip(y_final * _forecast_global_bias(), 0, capacity_mw)

    auto_winter_factor = feat.get("auto_winter_factor")
    if auto_winter_factor is None:
        auto_winter_factor = np.ones(len(feat), dtype=float)
        feat["auto_winter_factor"] = auto_winter_factor
    auto_winter_factor = np.asarray(auto_winter_factor, dtype=float)

    winter_factor = auto_winter_factor.copy()

    manual_factor_value = 1.0
    if manual_snow_enable and manual_snow_factor is not None:
        try:
            manual_factor_value = float(manual_snow_factor)
        except (TypeError, ValueError):
            manual_factor_value = 1.0
    manual_factor_value = float(np.clip(manual_factor_value, 0.0, MANUAL_SNOW_FACTOR_MAX))

    feat["manual_snow_factor"] = manual_factor_value
    manual_dates = manual_snow_dates or []
    if manual_snow_enable:
        if manual_dates:
            manual_mask = pd.to_datetime(feat["ds"]).dt.date.isin(manual_dates)
            winter_factor = winter_factor.copy()
            if manual_mask.any():
                winter_factor[manual_mask] = manual_factor_value
            else:
                winter_factor = np.full(len(feat), manual_factor_value, dtype=float)
        else:
            winter_factor = np.full(len(feat), manual_factor_value, dtype=float)

    feat["winter_factor_applied"] = winter_factor
    y_np = np.clip(y_np * winter_factor, 0, capacity_mw)
    y_xgb = np.clip(y_xgb * winter_factor, 0, capacity_mw)
    y_heur = np.clip(y_heur * winter_factor, 0, capacity_mw)
    y_final = np.clip(y_final * winter_factor, 0, capacity_mw)

    if _is_single_axis_tracker(st):
        if tracker_pvlib_pipeline:
            ac_cap_mw = _station_ac_nameplate_mw(st, capacity_mw)
            hist_cap_mw = _historical_tracker_output_cap_mw(st, ac_cap_mw)
            safe_cap_mw = float(np.clip(hist_cap_mw if hist_cap_mw is not None else ac_cap_mw, 0.0, ac_cap_mw))
            y_final = np.clip(y_final, 0.0, safe_cap_mw)
            tracker_caps = {"ac_cap_mw": float(ac_cap_mw), "safe_cap_mw": safe_cap_mw, "pvlib_pipeline": 1.0}
            tracker_postprocessing_applied = True
            logger.info("[FORECAST] station %s tracker pvlib cap applied: %s", st.pk, tracker_caps)
        else:
            y_final, tracker_caps = _apply_single_axis_tracker_postprocessing(y_final, feat, st, capacity_mw)
            tracker_postprocessing_applied = True
            logger.info(
                "[FORECAST] station %s single-axis tracker post-processing applied: %s",
                st.pk,
                tracker_caps,
            )
    else:
        y_final, early_morning_caps = _apply_early_morning_history_cap(y_final, feat, st, capacity_mw)
        if early_morning_caps:
            logger.info("[FORECAST] station %s early-morning history caps applied: %s", st.pk, early_morning_caps)

    if not tracker_pvlib_pipeline:
        y_final = _apply_tracker_midday_expected_floor(y_final, feat, st, capacity_mw)

    guardrail_df = pd.DataFrame(
        {
            "timestamp": feat["ds"],
            "irradiation": feat.get("irradiation", pd.Series([pd.NA] * len(feat))),
            "pred_final_mw": y_final,
        }
    )
    if abs(capacity_mw - 1.2) < 0.05:
        guardrail_df = apply_visual_crossing_fallback(guardrail_df)
        write_forecast_guardrail_log(guardrail_df)
        y_final = guardrail_df["pred_final_mw"].to_numpy(dtype=float)
    else:
        guardrail_df["pred_final_raw_mw"] = y_final
        guardrail_df["guardrail_reason"] = "OK"

    y_np_kw = y_np * 1000.0
    y_xgb_kw = y_xgb * 1000.0
    y_heur_kw = y_heur * 1000.0
    y_final_kw = y_final * 1000.0
    y_final_raw_kw = pd.to_numeric(guardrail_df["pred_final_raw_mw"], errors="coerce").to_numpy(dtype=float) * 1000.0

    # ---- save ----
    base_cleanup_start = timezone.datetime.combine(
        min(target_dates) if target_dates else start_date,
        timezone.datetime.min.time(),
    ).replace(tzinfo=now.tzinfo)
    base_cleanup_end = timezone.datetime.combine(
        (max(target_dates) + timedelta(days=1)) if target_dates else (start_date + timedelta(days=effective_days)),
        timezone.datetime.min.time(),
    ).replace(tzinfo=now.tzinfo)
    shifted_cleanup_start = base_cleanup_start + timedelta(hours=forecast_shift_hours)
    shifted_cleanup_end = base_cleanup_end + timedelta(hours=forecast_shift_hours)

    feat["ds"] = pd.to_datetime(feat["ds"], errors="coerce") + pd.to_timedelta(forecast_shift_hours, unit="h")
    logger.info(
        "[FORECAST_SHIFT] station=%s shift=%sh applied rows=%s",
        st.name,
        forecast_shift_hours,
        len(feat),
    )

    objs: List[SolarForecast] = []
    for i, row in feat.iterrows():
        pred_np_kw = None
        pred_xgb_kw = None
        if use_models:
            pred_np_kw = float(y_np_kw[i]) if not np.isnan(y_np_kw[i]) else None
            pred_xgb_kw = float(y_xgb_kw[i]) if not np.isnan(y_xgb_kw[i]) else None
        objs.append(
            SolarForecast(
                station=st,
                timestamp=pd.to_datetime(row["ds"]).to_pydatetime(),
                forecast_scope=forecast_scope,
                # Сохраняем в кВт (модель работает в MW, перевели выше)
                pred_np=pred_np_kw,
                pred_xgb=pred_xgb_kw,
                pred_heur=float(y_heur_kw[i]),
                pred_final=float(y_final_kw[i]),
                pred_final_raw=float(y_final_raw_kw[i]) if not np.isnan(y_final_raw_kw[i]) else None,
                guardrail_reason=str(guardrail_df.at[i, "guardrail_reason"] or "OK"),
                irradiation_fc=float(row.get("irradiation") or 0.0) if not pd.isna(row.get("irradiation")) else None,
                air_temp_fc=float(row.get("air_temp") or 0.0) if not pd.isna(row.get("air_temp")) else None,
                wind_speed_fc=float(row.get("wind_speed") or 0.0) if not pd.isna(row.get("wind_speed")) else None,
                cloudcover_fc=float(row.get("cloudcover") or 0.0) if not pd.isna(row.get("cloudcover")) else None,
                humidity_fc=float(row.get("humidity") or 0.0) if not pd.isna(row.get("humidity")) else None,
                precip_fc=float(row.get("precip") or 0.0) if not pd.isna(row.get("precip")) else None,
                snowfall_fc=float(row.get("snowfall") or 0.0) if not pd.isna(row.get("snowfall")) else None,
                snowdepth_fc=float(row.get("snowdepth") or 0.0) if not pd.isna(row.get("snowdepth")) else None,
                weather_code_fc=int(row.get("weather_code")) if not pd.isna(row.get("weather_code")) else None,
                auto_snow_flag=int(row.get("auto_snow_flag") or 0) if not pd.isna(row.get("auto_snow_flag")) else None,
                auto_fog_flag=int(row.get("auto_fog_flag") or 0) if not pd.isna(row.get("auto_fog_flag")) else None,
                auto_winter_factor=float(row.get("auto_winter_factor") or 1.0)
                if not pd.isna(row.get("auto_winter_factor"))
                else None,
                manual_snow_factor=float(row.get("manual_snow_factor") or 1.0)
                if not pd.isna(row.get("manual_snow_factor"))
                else None,
                winter_factor_applied=float(row.get("winter_factor_applied") or 1.0)
                if not pd.isna(row.get("winter_factor_applied"))
                else None,
            )
        )

    _replace_solar_forecast_rows_with_retry(
        station=st,
        forecast_scope=forecast_scope,
        cleanup_start=shifted_cleanup_start,
        cleanup_end=shifted_cleanup_end,
        objs=objs,
    )

    mirrored_qs = SolarForecast.objects.filter(
        station=st,
        forecast_scope=forecast_scope,
        timestamp__gte=shifted_cleanup_start,
        timestamp__lt=shifted_cleanup_end,
    ).order_by("id")
    sync_solar_forecasts(mirrored_qs)

    return {
        "ok": True,
        "count": len(objs),
        "days": report_days,
        "requested_days": days,
        "solar_hours": list(solar_hours),
        "weather_source": weather_source,
        "np_ok": np_ok,
        "xgb_ok": xgb_ok,
        "np_error": np_error,
        "xgb_error": xgb_error,
        "horizon_mode": horizon_mode,
        "target_dates": sorted(str(d) for d in (target_dates or [])),
        "forecast_scope": forecast_scope,
        "mount_type": getattr(st, "mount_type", Station.MOUNT_FIXED),
        "tracker_postprocessing_applied": tracker_postprocessing_applied,
        "tracker_caps": tracker_caps,
    }
