"""Training and prediction helpers for single-axis tracker PV stations.

Tracker stations must train and predict from the same physical input chain used
in production weather forecasts:

    GHI -> pvlib single-axis GHI-to-POA -> ML model -> power MW

The measured historical POA column is intentionally kept out of model inputs and
is exported only as a diagnostic for the pvlib converter quality.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
import neuralprophet
from neuralprophet import NeuralProphet, save as np_save

from stations.models import Station
from .model_storage import resolve_station_model_dir
from .train_models import (
    MODEL_DIR,
    PR_FOR_EXPECTED,
    add_common_features,
    add_sun_geometry,
    get_history_dataframe,
    station_capacity_mw,
)


TRACKER_XGB_FEATURES = [
    "Irradiation_GHI",
    "POA_pvlib",
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
    "tracker_theta",
    "tracker_aoi",
    "tracker_surface_tilt",
    "tracker_surface_azimuth",
    "tracker_pvlib_baseline_mw",
    "tracker_pvlib_baseline_log",
]

TRACKER_NP_REGRESSORS = [
    "Irradiation_GHI",
    "POA_pvlib",
    "Air_Temp",
    "PV_Temp",
    "Wind_Speed",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "is_clear",
    "sun_elev_deg",
    "low_sun_flag",
    "sunrise_hour_flag",
    "solar_ramp_factor",
    "irradiation_x_ramp",
    "tracker_theta",
    "tracker_aoi",
    "tracker_surface_tilt",
    "tracker_surface_azimuth",
    "tracker_pvlib_baseline_mw",
    "tracker_pvlib_baseline_log",
]


@dataclass(frozen=True)
class TrackerPvlibConfig:
    latitude: float
    longitude: float
    timezone: str
    axis_tilt: float = 0.0
    axis_azimuth: float = 0.0
    max_angle: float = 30.0
    gcr: float = 0.40
    backtrack: bool = True
    model: str = "perez"
    albedo: float = 0.20

    def as_meta(self) -> dict[str, Any]:
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "timezone": self.timezone,
            "axis_tilt": self.axis_tilt,
            "axis_azimuth": self.axis_azimuth,
            "max_angle": self.max_angle,
            "gcr": self.gcr,
            "backtrack": self.backtrack,
            "model": self.model,
            "albedo": self.albedo,
        }


def is_single_axis_tracker(station: Station) -> bool:
    mount_type = str(getattr(station, "mount_type", Station.MOUNT_FIXED) or Station.MOUNT_FIXED)
    return mount_type.strip().lower().replace("-", "_") == Station.MOUNT_SINGLE_AXIS_TRACKER


def tracker_config_from_station(station: Station) -> TrackerPvlibConfig:
    lat = getattr(station, "latitude", None)
    lon = getattr(station, "longitude", None)
    if lat is None or lon is None:
        raise ValueError("Tracker pvlib training requires station latitude and longitude")

    def as_float(name: str, default: float) -> float:
        raw = getattr(station, name, default)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = default
        return value

    return TrackerPvlibConfig(
        latitude=float(lat),
        longitude=float(lon),
        timezone=str(getattr(station, "timezone", None) or "Asia/Almaty"),
        axis_tilt=as_float("tracker_axis_tilt", 0.0),
        axis_azimuth=as_float("tracker_axis_azimuth", 0.0),
        max_angle=as_float("tracker_max_angle", 30.0),
        gcr=as_float("tracker_gcr", 0.40),
        backtrack=bool(getattr(station, "tracker_backtrack", True)),
        model=str(getattr(station, "tracker_poa_model", None) or "perez"),
        albedo=as_float("tracker_albedo", 0.20),
    )


def _localized_index(ds: pd.Series, tz_name: str) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(ds, errors="coerce"))
    if idx.tz is None:
        return idx.tz_localize(tz_name, nonexistent="shift_forward", ambiguous="NaT")
    return idx.tz_convert(tz_name)


def add_pvlib_tracker_poa(
    df: pd.DataFrame,
    station: Station,
    *,
    ghi_col: str = "Irradiation_GHI",
    output_col: str = "POA_pvlib",
) -> pd.DataFrame:
    """Add pvlib-computed tracker POA and tracker geometry columns.

    Only GHI is used as irradiance input. Historical measured POA is not read by
    this function and therefore cannot leak into model features.
    """
    import pvlib

    cfg = tracker_config_from_station(station)
    out = df.copy()
    if out.empty:
        out[output_col] = []
        return out

    times = _localized_index(out["ds"], cfg.timezone)
    ghi = pd.to_numeric(out.get(ghi_col), errors="coerce").fillna(0.0).clip(lower=0.0)
    ghi.index = times

    solar_position = pvlib.solarposition.get_solarposition(times, cfg.latitude, cfg.longitude)
    zenith = solar_position["apparent_zenith"].clip(lower=0.0, upper=180.0)
    azimuth = solar_position["azimuth"]

    erbs = pvlib.irradiance.erbs(ghi, zenith, times)
    dni = pd.to_numeric(erbs["dni"], errors="coerce").fillna(0.0).clip(lower=0.0)
    dhi = pd.to_numeric(erbs["dhi"], errors="coerce").fillna(0.0).clip(lower=0.0)

    tracking = pvlib.tracking.singleaxis(
        apparent_zenith=zenith,
        apparent_azimuth=azimuth,
        axis_tilt=cfg.axis_tilt,
        axis_azimuth=cfg.axis_azimuth,
        max_angle=cfg.max_angle,
        backtrack=cfg.backtrack,
        gcr=cfg.gcr,
    ).fillna(0.0)

    dni_extra = pvlib.irradiance.get_extra_radiation(times)
    airmass = pvlib.atmosphere.get_relative_airmass(zenith)
    total = pvlib.irradiance.get_total_irradiance(
        surface_tilt=tracking["surface_tilt"],
        surface_azimuth=tracking["surface_azimuth"],
        dni=dni,
        ghi=ghi,
        dhi=dhi,
        solar_zenith=zenith,
        solar_azimuth=azimuth,
        dni_extra=dni_extra,
        airmass=airmass,
        model=cfg.model,
        albedo=cfg.albedo,
    )

    poa = pd.to_numeric(total["poa_global"], errors="coerce").fillna(0.0).clip(lower=0.0)
    out[output_col] = poa.to_numpy(dtype=float)
    out["DNI_erbs"] = dni.to_numpy(dtype=float)
    out["DHI_erbs"] = dhi.to_numpy(dtype=float)
    out["tracker_theta"] = pd.to_numeric(tracking.get("tracker_theta"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    out["tracker_aoi"] = pd.to_numeric(tracking.get("aoi"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    out["tracker_surface_tilt"] = pd.to_numeric(tracking.get("surface_tilt"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    out["tracker_surface_azimuth"] = pd.to_numeric(tracking.get("surface_azimuth"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return out


def _add_tracker_baseline(df: pd.DataFrame, cap_mw: float, pr: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = df.copy()
    poa = pd.to_numeric(out["POA_pvlib"], errors="coerce").fillna(0.0).clip(lower=0.0)
    base = (cap_mw * (poa / 1000.0) * pr).clip(lower=0.0, upper=cap_mw)
    y = pd.to_numeric(out.get("y"), errors="coerce").fillna(0.0)

    denom = base.clip(lower=max(0.03 * cap_mw, 0.05))
    ratio = (y / denom).replace([np.inf, -np.inf], np.nan)
    valid = (poa >= 80.0) & ratio.notna() & ratio.between(0.2, 2.0)
    bins = [0, 100, 200, 350, 500, 650, 800, 1000, 1400]
    labels = [f"{bins[i]}-{bins[i + 1]}" for i in range(len(bins) - 1)]
    poa_bin = pd.cut(poa, bins=bins, labels=labels, include_lowest=True, right=False)
    curve = ratio.loc[valid].groupby(poa_bin.loc[valid], observed=True).median().clip(0.70, 1.30).to_dict()
    factor = poa_bin.map(curve).astype(float).fillna(1.0)

    out["tracker_pvlib_baseline_mw"] = (base * factor).clip(lower=0.0, upper=cap_mw)
    out["tracker_pvlib_baseline_log"] = np.log1p(out["tracker_pvlib_baseline_mw"])
    meta = {
        "type": "poa_bin_median_ratio",
        "bins": bins,
        "curve": {str(k): float(v) for k, v in curve.items()},
        "pr": float(pr),
    }
    return out, meta


def apply_tracker_baseline_from_meta(df: pd.DataFrame, cap_mw: float, meta: dict[str, Any] | None = None) -> pd.DataFrame:
    out = df.copy()
    baseline_meta = (meta or {}).get("baseline_curve") if isinstance(meta, dict) else {}
    pr = float((baseline_meta or {}).get("pr", PR_FOR_EXPECTED))
    bins = (baseline_meta or {}).get("bins") or [0, 100, 200, 350, 500, 650, 800, 1000, 1400]
    curve = (baseline_meta or {}).get("curve") or {}

    poa = pd.to_numeric(out.get("POA_pvlib"), errors="coerce").fillna(0.0).clip(lower=0.0)
    base = (cap_mw * (poa / 1000.0) * pr).clip(lower=0.0, upper=cap_mw)
    labels = [f"{bins[i]}-{bins[i + 1]}" for i in range(len(bins) - 1)]
    poa_bin = pd.cut(poa, bins=bins, labels=labels, include_lowest=True, right=False)
    factor = poa_bin.astype(str).map({str(k): float(v) for k, v in curve.items()}).fillna(1.0)
    out["tracker_pvlib_baseline_mw"] = (base * factor).clip(lower=0.0, upper=cap_mw)
    out["tracker_pvlib_baseline_log"] = np.log1p(out["tracker_pvlib_baseline_mw"])
    return out


def add_tracker_prediction_features(
    df: pd.DataFrame,
    station: Station,
    capacity_mw: float,
    meta: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Add operational prediction features from forecast GHI via pvlib."""
    out = df.copy()
    out["Irradiation_GHI"] = pd.to_numeric(out.get("Irradiation"), errors="coerce").fillna(0.0)
    out = add_pvlib_tracker_poa(out, station, ghi_col="Irradiation_GHI")
    out = apply_tracker_baseline_from_meta(out, capacity_mw, meta)
    return out


def _tracker_history_dataframe(station: Station) -> pd.DataFrame:
    df = get_history_dataframe(station)
    if df.empty:
        return df
    df["Irradiation_GHI"] = pd.to_numeric(df.get("Irradiation_GHI"), errors="coerce")
    df["POA_real_diagnostic"] = pd.to_numeric(df.get("Irradiation_POA"), errors="coerce")
    df["Air_Temp"] = pd.to_numeric(df.get("Air_Temp"), errors="coerce")
    df["Power_KW"] = pd.to_numeric(df.get("Power_KW"), errors="coerce")
    before = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["ds", "Power_KW", "Irradiation_GHI", "Air_Temp"])
    removed = before - len(df)
    if removed:
        print(f"[TRACKER_PVLIB_TRAIN] station {station.pk}: removed {removed} rows without GHI/power/air_temp")
    return df


def _write_backtest_excel(df: pd.DataFrame, model_dir: Path) -> Path | None:
    cols = [
        "ds",
        "Power_KW",
        "Irradiation_GHI",
        "POA_real_diagnostic",
        "POA_pvlib",
        "POA_pvlib_error",
        "POA_pvlib_abs_error",
        "tracker_pvlib_baseline_mw",
        "xgb_pred_mw",
    ]
    available = [c for c in cols if c in df.columns]
    if not available:
        return None
    path = model_dir / "tracker_pvlib_backtest.xlsx"
    export_df = df[available].copy()
    if "ds" in export_df.columns:
        ds = pd.to_datetime(export_df["ds"], errors="coerce")
        try:
            if getattr(ds.dt, "tz", None) is not None:
                ds = ds.dt.tz_localize(None)
        except (AttributeError, TypeError):
            pass
        export_df["ds"] = ds
    try:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            export_df.to_excel(writer, index=False, sheet_name="backtest")
    except Exception as exc:
        # Backtest export must not make the whole training command fail after
        # models were already fitted/saved.  A previous run crashed here on
        # tz-aware datetimes, leaving the UI looking like XGBoost was not
        # retrained even though training had reached the end.
        print(f"[TRACKER_PVLIB_TRAIN] backtest Excel export skipped: {exc}")
        return None
    return path


def train_tracker_pvlib_models_for_station(station: Station) -> Tuple[int, Path | None, Path | None]:
    if not is_single_axis_tracker(station):
        raise ValueError("tracker_pvlib_training can run only for single-axis tracker stations")

    df = _tracker_history_dataframe(station)
    if df.empty:
        return 0, None, None

    cap_mw = station_capacity_mw(station, df)
    pr = float(getattr(station, "pr_default", PR_FOR_EXPECTED) or PR_FOR_EXPECTED)
    pr = float(np.clip(pr, 0.10, 1.00))
    lat = float(getattr(station, "latitude", 47.86) or 47.86)

    df["y"] = (df["Power_KW"] / 1000.0).clip(lower=0.0)
    df["Irradiation"] = df["Irradiation_GHI"]
    df = add_common_features(df, cap_mw, "ds")
    df = add_sun_geometry(df, "ds", lat)
    df = add_pvlib_tracker_poa(df, station, ghi_col="Irradiation_GHI")
    df, baseline_meta = _add_tracker_baseline(df, cap_mw, pr)

    df["POA_pvlib_error"] = df["POA_pvlib"] - df["POA_real_diagnostic"]
    df["POA_pvlib_abs_error"] = df["POA_pvlib_error"].abs()

    model_dir = resolve_station_model_dir(MODEL_DIR, station, create=True)
    n_rows = len(df)

    xgb_path: Path | None = None
    np_path: Path | None = None
    xgb_meta: dict[str, Any] = {}

    train_df = df.copy().replace([np.inf, -np.inf], np.nan)
    for col in TRACKER_XGB_FEATURES:
        if col not in train_df.columns:
            train_df[col] = np.nan
    train_df = train_df.dropna(subset=["y"] + TRACKER_XGB_FEATURES)
    train_df = train_df[(train_df["POA_pvlib"] > 20.0) | (train_df["y"] > 0.02)]

    if not train_df.empty:
        model_xgb = xgb.XGBRegressor(
            n_estimators=900,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            min_child_weight=3,
            random_state=42,
        )
        sample_weight = 1.0 + 3.0 * np.clip(train_df["POA_pvlib"] / 800.0, 0.0, 1.0)
        model_xgb.fit(train_df[TRACKER_XGB_FEATURES], train_df["y"], sample_weight=sample_weight)
        xgb_pred = model_xgb.predict(df[TRACKER_XGB_FEATURES].fillna(0.0))
        df["xgb_pred_mw"] = np.clip(xgb_pred, 0.0, cap_mw)
        xgb_path = model_dir / "xgb_model.json"
        model_xgb.save_model(str(xgb_path))
        xgb_meta = {
            "station_id": station.pk,
            "pipeline": "tracker_pvlib_v1",
            "X_cols": TRACKER_XGB_FEATURES,
            "cap_mw_used": cap_mw,
            "target": "direct_mw",
            "input_chain": "GHI -> pvlib singleaxis -> POA_pvlib -> XGBoost -> MW",
            "poa_real_usage": "diagnostic_only_not_model_input",
            "baseline_curve": baseline_meta,
            "pvlib": tracker_config_from_station(station).as_meta(),
            "train_rows_total": int(n_rows),
            "train_rows_used": int(len(train_df)),
            "xgb_version": getattr(xgb, "__version__", "unknown"),
        }
        (model_dir / "xgb_model.meta.json").write_text(json.dumps(xgb_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    try:
        df_np = df.copy()
        df_np["y_residual"] = df_np["y"] - pd.to_numeric(df_np["tracker_pvlib_baseline_mw"], errors="coerce").fillna(0.0)
        m = NeuralProphet(
            n_lags=0,
            n_forecasts=1,
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=True,
            seasonality_mode="additive",
            learning_rate=0.2,
            epochs=450,
            batch_size=64,
            loss_func="MSE",
        )
        for col in TRACKER_NP_REGRESSORS:
            m.add_future_regressor(col, normalize="minmax")

        df_fit = df_np[["ds", "y_residual"] + TRACKER_NP_REGRESSORS].copy().rename(columns={"y_residual": "y"})
        fill_map: dict[str, float] = {}
        for col in TRACKER_NP_REGRESSORS:
            df_fit[col] = pd.to_numeric(df_fit[col], errors="coerce")
            med = df_fit[col].median(skipna=True)
            fill_map[col] = float(0.0 if pd.isna(med) else med)
            df_fit[col] = df_fit[col].fillna(fill_map[col])
        df_fit = df_fit.dropna(subset=["ds", "y"])
        if len(df_fit) < 500:
            raise RuntimeError(f"Too few rows for tracker NP after cleaning: {len(df_fit)}")
        m.fit(df_fit, freq="h")
        np_path = model_dir / "np_model.np"
        if hasattr(m, "save"):
            m.save(str(np_path))
        else:
            np_save(m, str(np_path))
        np_meta = {
            "station_id": station.pk,
            "pipeline": "tracker_pvlib_v1",
            "cap_mw": cap_mw,
            "features_reg": TRACKER_NP_REGRESSORS,
            "fill_map": fill_map,
            "target": "y_residual = y - tracker_pvlib_baseline_mw",
            "base_expected_col": "tracker_pvlib_baseline_mw",
            "input_chain": "GHI -> pvlib singleaxis -> POA_pvlib -> NeuralProphet residual -> MW",
            "poa_real_usage": "diagnostic_only_not_model_input",
            "baseline_curve": baseline_meta,
            "pvlib": tracker_config_from_station(station).as_meta(),
            "np_version": getattr(neuralprophet, "__version__", "unknown"),
        }
        (model_dir / "np_model.meta.json").write_text(json.dumps(np_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        import traceback

        print(f"[TRACKER_PVLIB_TRAIN] station {station.pk}: NP failed -> {exc}")
        traceback.print_exc()
        np_path = None

    ensemble_meta = {
        "station_id": station.pk,
        "pipeline": "tracker_pvlib_v1",
        "input_chain": "historical/forecast GHI -> pvlib GHI→POA -> baseline + NP residual + XGB direct -> Power MW",
        "poa_real_usage": "backtest_diagnostic_only",
        "cap_mw": cap_mw,
        "baseline_curve": baseline_meta,
        "pvlib": tracker_config_from_station(station).as_meta(),
        "models": {
            "np": str(np_path) if np_path else None,
            "xgb": str(xgb_path) if xgb_path else None,
        },
    }
    (model_dir / "ensemble.meta.json").write_text(json.dumps(ensemble_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_backtest_excel(df, model_dir)

    return n_rows, np_path, xgb_path
