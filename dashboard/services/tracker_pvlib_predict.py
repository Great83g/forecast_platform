"""Operational PVLIB POA predict pipeline for single-axis tracker stations."""
from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from django.conf import settings

from solar.models import SolarRecord
from stations.models import Station
from .model_storage import resolve_station_model_dir
from .tracker_pvlib_training import (
    TRACKER_NP_REGRESSORS,
    TRACKER_XGB_FEATURES,
    add_tracker_prediction_features,
    tracker_config_from_station,
)

logger = logging.getLogger(__name__)
MODEL_DIR: Path = Path(getattr(settings, "MODEL_DIR", Path(settings.BASE_DIR) / "models_cache"))


@dataclass
class TrackerPredictionResult:
    feat: pd.DataFrame
    y_np_mwh: np.ndarray
    y_xgb_mwh: np.ndarray
    y_ensemble_base_mwh: np.ndarray
    hist_analog_mwh: np.ndarray
    y_final_mwh: np.ndarray
    np_ok: bool
    xgb_ok: bool
    method: str
    errors: dict[str, str]


def tracker_model_paths(station: Station) -> dict[str, Path]:
    model_dir = resolve_station_model_dir(MODEL_DIR, station)
    return {
        "model_dir": model_dir,
        "np_named": model_dir / "trained_tracker_100mw_np_PVLIB_POA.np",
        "xgb_named": model_dir / "trained_tracker_100mw_xgb_PVLIB_POA.pkl",
        "meta_named": model_dir / "trained_tracker_100mw_PVLIB_POA_ENSEMBLE.meta.json",
        "np": model_dir / "np_model.np",
        "xgb_json": model_dir / "xgb_model.json",
        "xgb_meta": model_dir / "xgb_model.meta.json",
        "np_meta": model_dir / "np_model.meta.json",
        "ensemble_meta": model_dir / "ensemble.meta.json",
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.exception("[TRACKER_PREDICT] failed to read meta %s", path)
    return {}


def _load_np(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        from neuralprophet import load as np_load

        return np_load(str(path))
    except Exception:
        logger.exception("[TRACKER_PREDICT] failed to load NP %s", path)
        return None


def _load_xgb(path: Path) -> tuple[Any | None, str]:
    if not path.exists():
        return None, ""
    try:
        if path.suffix.lower() == ".pkl":
            try:
                import joblib

                return joblib.load(path), "sklearn"
            except Exception:
                with path.open("rb") as fh:
                    return pickle.load(fh), "sklearn"
        import xgboost as xgb

        booster = xgb.Booster()
        booster.load_model(str(path))
        return booster, "booster"
    except Exception:
        logger.exception("[TRACKER_PREDICT] failed to load XGB %s", path)
        return None, ""


def _predict_np(model: Any, feat: pd.DataFrame, reg_cols: list[str], fill_map: dict[str, Any]) -> np.ndarray:
    dfp = pd.DataFrame({"ds": pd.to_datetime(feat["ds"]), "y": np.nan})
    for col in reg_cols:
        default = fill_map.get(col, 0.0) if isinstance(fill_map, dict) else 0.0
        if col in feat.columns:
            series = pd.to_numeric(feat[col], errors="coerce")
        else:
            series = pd.Series(default, index=feat.index)
        dfp[col] = series.fillna(float(default)).to_numpy()
    fcst = model.predict(dfp)
    yhat_col = "yhat1" if "yhat1" in fcst.columns else next((c for c in fcst.columns if c.startswith("yhat")), None)
    if yhat_col is None:
        return np.full(len(feat), np.nan)
    yhat = pd.to_numeric(fcst[yhat_col], errors="coerce")
    if len(yhat) == len(feat):
        return yhat.to_numpy(dtype=float)
    aligned = (
        pd.DataFrame({"ds": pd.to_datetime(dfp["ds"]).dt.floor("h")})
        .merge(
            pd.DataFrame({"ds": pd.to_datetime(fcst["ds"], errors="coerce").dt.floor("h"), "yhat": yhat}).dropna(subset=["ds"]),
            on="ds",
            how="left",
        )["yhat"]
    )
    return pd.to_numeric(aligned, errors="coerce").to_numpy(dtype=float)


def _predict_xgb(model: Any, model_kind: str, feat: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    X = feat.copy()
    for col in feature_cols:
        if col not in X.columns:
            X[col] = 0.0
    X = X[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    if model_kind == "booster":
        import xgboost as xgb

        return np.asarray(model.predict(xgb.DMatrix(X, feature_names=feature_cols)), dtype=float)
    return np.asarray(model.predict(X), dtype=float)


def convert_ghi_to_poa_pvlib(df: pd.DataFrame, station: Station, capacity_mw: float, meta: dict[str, Any] | None = None) -> pd.DataFrame:
    """Convert Visual Crossing GHI to pvlib tracker POA and baseline features."""
    out = add_tracker_prediction_features(df, station, capacity_mw, meta or {})
    if "tracker_surface_tilt" in out.columns:
        out["tracker_tilt"] = out["tracker_surface_tilt"]
    if "tracker_surface_azimuth" in out.columns:
        out["tracker_azimuth"] = out["tracker_surface_azimuth"]
    return out


def add_features(df: pd.DataFrame, station: Station, capacity_mw: float, meta: dict[str, Any] | None = None) -> pd.DataFrame:
    """Ensure PVLIB POA ensemble feature columns are present for prediction."""
    out = convert_ghi_to_poa_pvlib(df, station, capacity_mw, meta)
    for col in set(TRACKER_XGB_FEATURES + TRACKER_NP_REGRESSORS):
        if col not in out.columns:
            out[col] = 0.0
    return out


def historical_analog_pvlib_poa(feat: pd.DataFrame, station: Station, capacity_mw: float) -> np.ndarray:
    """POA/hour/month historical analog in MWh for one-hour rows."""
    rows = list(
        SolarRecord.objects.filter(
            station=station,
            history_scope=SolarRecord.HISTORY_SCOPE_MAIN,
            power_kw__isnull=False,
            irradiation_ghi__isnull=False,
        )
        .order_by("-timestamp")
        .values("timestamp", "power_kw", "irradiation_ghi", "air_temp")[:24 * 365]
    )
    if len(rows) < 24:
        return np.full(len(feat), np.nan)
    hist = pd.DataFrame(rows).rename(columns={"timestamp": "ds", "irradiation_ghi": "Irradiation_GHI", "air_temp": "Air_Temp"})
    hist["Irradiation"] = pd.to_numeric(hist["Irradiation_GHI"], errors="coerce")
    hist["air_temp"] = pd.to_numeric(hist["Air_Temp"], errors="coerce")
    try:
        hist_feat = add_tracker_prediction_features(hist, station, capacity_mw, {})
    except Exception:
        logger.exception("[TRACKER_PREDICT] failed to build historical analog POA")
        return np.full(len(feat), np.nan)
    hist_feat["hour"] = pd.to_datetime(hist_feat["ds"]).dt.hour.astype(int)
    hist_feat["month"] = pd.to_datetime(hist_feat["ds"]).dt.month.astype(int)
    hist_feat["power_mwh"] = (pd.to_numeric(hist_feat["power_kw"], errors="coerce") / 1000.0).clip(0.0, capacity_mw)
    hist_feat["poa_bin"] = pd.cut(pd.to_numeric(hist_feat["POA_pvlib"], errors="coerce"), bins=[0, 50, 100, 200, 350, 500, 650, 800, 1000, 1400], include_lowest=True, right=False)
    table = hist_feat.dropna(subset=["hour", "month", "poa_bin", "power_mwh"]).groupby(["month", "hour", "poa_bin"], observed=True)["power_mwh"].median()
    out = []
    f = feat.copy()
    f["hour"] = pd.to_datetime(f["ds"]).dt.hour.astype(int)
    f["month"] = pd.to_datetime(f["ds"]).dt.month.astype(int)
    f["poa_bin"] = pd.cut(pd.to_numeric(f["POA_pvlib"], errors="coerce"), bins=[0, 50, 100, 200, 350, 500, 650, 800, 1000, 1400], include_lowest=True, right=False)
    for row in f.itertuples(index=False):
        val = table.get((int(row.month), int(row.hour), row.poa_bin), np.nan)
        if pd.isna(val):
            hour_vals = hist_feat.loc[hist_feat["hour"] == int(row.hour), "power_mwh"].dropna()
            val = float(hour_vals.median()) if len(hour_vals) else np.nan
        out.append(val)
    return np.asarray(out, dtype=float)


def postprocess(
    feat: pd.DataFrame,
    station: Station,
    capacity_mw: float,
    y_np_mwh: np.ndarray,
    y_xgb_mwh: np.ndarray,
    np_ok: bool,
    xgb_ok: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PVLIB tracker ensemble postprocess with analog, evening, morning and night fixes."""
    baseline = pd.to_numeric(feat.get("tracker_pvlib_baseline_mw"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if np_ok and xgb_ok:
        ensemble_base = 0.45 * np.nan_to_num(y_np_mwh, nan=0.0) + 0.55 * np.nan_to_num(y_xgb_mwh, nan=0.0)
    elif xgb_ok:
        ensemble_base = np.nan_to_num(y_xgb_mwh, nan=0.0)
    elif np_ok:
        ensemble_base = np.nan_to_num(y_np_mwh, nan=0.0)
    else:
        ensemble_base = baseline.copy()

    hist = historical_analog_pvlib_poa(feat, station, capacity_mw)
    has_hist = np.isfinite(hist)
    final = ensemble_base.copy()
    final[has_hist] = 0.85 * final[has_hist] + 0.15 * hist[has_hist]

    hours = pd.to_datetime(feat["ds"]).dt.hour.astype(int).to_numpy()
    poa = pd.to_numeric(feat.get("POA_pvlib"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    sun_elev = pd.to_numeric(feat.get("sun_elev_deg"), errors="coerce").fillna(0.0).to_numpy(dtype=float)

    # Daytime physical recovery: a bad/out-of-domain ML artifact must not collapse
    # a clear tracker day to near-zero while pvlib POA is high.  The screenshot
    # failure mode was exactly that: good GHI/POA at 12:00+, but ML output went
    # almost to zero.  Keep the learned model, but enforce a conservative floor
    # from the pvlib baseline for strong daylight hours.
    strong_daylight = (hours >= 8) & (hours <= 16) & (poa >= 180.0)
    daylight_floor = baseline * 0.90
    high_poa_floor = baseline * 0.96
    final[strong_daylight] = np.maximum(final[strong_daylight], daylight_floor[strong_daylight])
    high_poa_daylight = strong_daylight & (poa >= 500.0)
    final[high_poa_daylight] = np.maximum(final[high_poa_daylight], high_poa_floor[high_poa_daylight])

    # Trackers have strong shoulders: for Shu-like single-axis plants 06-07 and
    # 17-18 can already produce a large share of AC power.  Do not let the
    # generic ML/fallback curve flatten those shoulders when GHI/POA says the
    # sun is available.  Ratios are conservative minimums and still clipped by
    # station AC capacity below.
    ghi = pd.to_numeric(
        feat.get("Irradiation_GHI", feat.get("Irradiation", pd.Series(0.0, index=feat.index))),
        errors="coerce",
    ).fillna(0.0).to_numpy(dtype=float)
    shoulder_ratio_by_hour = {6: 0.16, 7: 0.45, 8: 0.62, 17: 0.68, 18: 0.40, 19: 0.04}
    shoulder_min_ghi_by_hour = {6: 45.0, 7: 90.0, 8: 160.0, 17: 220.0, 18: 100.0, 19: 35.0}
    shoulder_min_poa_by_hour = {6: 20.0, 7: 45.0, 8: 90.0, 17: 100.0, 18: 45.0, 19: 15.0}
    for hour, ratio in shoulder_ratio_by_hour.items():
        mask = (
            (hours == hour)
            & (sun_elev > -2.0)
            & (
                (ghi >= shoulder_min_ghi_by_hour[hour])
                | (poa >= shoulder_min_poa_by_hour[hour])
                | (baseline >= capacity_mw * ratio * 0.45)
            )
        )
        final[mask] = np.maximum(final[mask], capacity_mw * ratio)

    # Extra morning recovery 08-11: do not let ML suppress clear tracker mornings too much.
    morning = np.isin(hours, [8, 9, 10, 11]) & (poa >= 120.0)
    final[morning] = np.maximum(final[morning], baseline[morning] * 0.92)

    # If a model is globally broken for the day (many high-POA hours below the
    # physical curve), distrust it and use the pvlib baseline blended with analog.
    high_poa = (hours >= 8) & (hours <= 16) & (poa >= 300.0) & (baseline > 0.0)
    if np.count_nonzero(high_poa) >= 3:
        collapse_ratio = np.count_nonzero(final[high_poa] < baseline[high_poa] * 0.35) / np.count_nonzero(high_poa)
        if collapse_ratio >= 0.30:
            guarded = baseline.copy()
            guarded[has_hist] = 0.85 * guarded[has_hist] + 0.15 * hist[has_hist]
            final[high_poa] = np.maximum(final[high_poa], guarded[high_poa] * 0.98)

    # adaptive evening fix: suppress late-day spikes when POA is falling/low.
    evening = (hours >= 17) & (hours <= 20)
    evening_cap = np.maximum(baseline * 1.12, np.where(has_hist, hist * 1.10, 0.0))
    final[evening] = np.minimum(final[evening], evening_cap[evening])

    # night mask.
    night = (poa <= 10.0) | (sun_elev <= 0.0) | (hours <= 4) | (hours >= 22)
    final[night] = 0.0
    final = np.clip(np.nan_to_num(final, nan=0.0), 0.0, capacity_mw)
    return final, ensemble_base, hist


def run_tracker_pvlib_predict(feat: pd.DataFrame, station: Station, capacity_mw: float, use_models: bool = True) -> TrackerPredictionResult:
    paths = tracker_model_paths(station)
    meta = _read_json(paths["meta_named"]) or _read_json(paths["ensemble_meta"]) or _read_json(paths["xgb_meta"]) or _read_json(paths["np_meta"])
    feat = add_features(feat, station, capacity_mw, meta).reset_index(drop=True)
    errors: dict[str, str] = {}

    np_path = paths["np_named"] if paths["np_named"].exists() else paths["np"]
    xgb_path = paths["xgb_named"] if paths["xgb_named"].exists() else paths["xgb_json"]

    y_np = np.full(len(feat), np.nan)
    y_xgb = np.full(len(feat), np.nan)
    np_ok = False
    xgb_ok = False

    if use_models:
        np_model = _load_np(np_path)
        if np_model is not None:
            try:
                np_meta = _read_json(paths["np_meta"]) or meta
                regs = np_meta.get("features_reg") or TRACKER_NP_REGRESSORS
                residual = _predict_np(np_model, feat, regs, np_meta.get("fill_map") or {})
                base_col = str(np_meta.get("base_expected_col") or "tracker_pvlib_baseline_mw")
                base = pd.to_numeric(feat.get(base_col, feat.get("tracker_pvlib_baseline_mw")), errors="coerce").fillna(0.0).to_numpy(dtype=float)
                y_np = np.clip(base + np.nan_to_num(residual, nan=0.0), 0.0, capacity_mw)
                np_ok = True
            except Exception as exc:
                errors["np"] = str(exc)
                logger.exception("[TRACKER_PREDICT] NP predict failed")
        else:
            errors["np"] = f"NP not found: {np_path}"

        xgb_model, xgb_kind = _load_xgb(xgb_path)
        if xgb_model is not None:
            try:
                xgb_meta = _read_json(paths["xgb_meta"]) or meta
                cols = xgb_meta.get("X_cols") or meta.get("X_cols") or TRACKER_XGB_FEATURES
                y_xgb = np.clip(_predict_xgb(xgb_model, xgb_kind, feat, cols), 0.0, capacity_mw)
                xgb_ok = True
            except Exception as exc:
                errors["xgb"] = str(exc)
                logger.exception("[TRACKER_PREDICT] XGB predict failed")
        else:
            errors["xgb"] = f"XGB not found: {xgb_path}"

    y_final, y_ensemble_base, hist = postprocess(feat, station, capacity_mw, y_np, y_xgb, np_ok, xgb_ok)
    method = "pvlib_tracker_ml" if (np_ok or xgb_ok) else "pvlib_tracker_fallback"
    return TrackerPredictionResult(
        feat=feat,
        y_np_mwh=y_np,
        y_xgb_mwh=y_xgb,
        y_ensemble_base_mwh=y_ensemble_base,
        hist_analog_mwh=hist,
        y_final_mwh=y_final,
        np_ok=np_ok,
        xgb_ok=xgb_ok,
        method=method,
        errors=errors,
    )
