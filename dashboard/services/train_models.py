# dashboard/services/train_models.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

from django.conf import settings
import numpy as np
import pandas as pd
import xgboost as xgb
import neuralprophet
from neuralprophet import NeuralProphet, save as np_save

from solar.models import SolarRecord
from stations.models import Station
from .model_storage import resolve_station_model_dir


def _capacity_mw_from_fields(station) -> float | None:
    capacity_ac_kw = None
    for name in ["capacity_ac_kw", "capacity_kw", "capacity_dc_kw"]:
        if hasattr(station, name) and getattr(station, name):
            capacity_ac_kw = float(getattr(station, name))
            break

    for name in ["capacity_mw", "capacity_ac_mw"]:
        if hasattr(station, name) and getattr(station, name):
            capacity_mw = float(getattr(station, name))
            if capacity_mw >= 1000:
                return capacity_mw / 1000.0
            if capacity_mw > 100 and capacity_ac_kw:
                return capacity_mw / 1000.0
            return capacity_mw

    if capacity_ac_kw:
        return capacity_ac_kw / 1000.0

    return None


def _history_scale_factor(target, source) -> float | None:
    if not getattr(target, "history_scale_by_capacity", False):
        return None

    target_cap = _capacity_mw_from_fields(target)
    source_cap = _capacity_mw_from_fields(source)
    if not target_cap or not source_cap:
        return None

    return target_cap / source_cap

# Папка с моделями (в backend.settings: MODEL_DIR = BASE_DIR / "models_cache")
MODEL_DIR: Path = Path(settings.MODEL_DIR)

# Параметры expected/PR
PR_FOR_EXPECTED = 0.90


def _station_model_dir(station) -> Path:
    return resolve_station_model_dir(MODEL_DIR, station, create=True)


def get_history_dataframe(station) -> pd.DataFrame:
    """
    История по станции из SolarRecord.

    Возвращает DataFrame с колонками:
      ds, Power_KW, Irradiation, Air_Temp, PV_Temp
    """
    history_station = station
    qs = SolarRecord.objects.filter(station=station, history_scope=SolarRecord.HISTORY_SCOPE_MAIN).order_by("timestamp")
    if not qs.exists() and getattr(station, "history_source_id", None):
        history_station = station.history_source
        qs = SolarRecord.objects.filter(station=history_station, history_scope=SolarRecord.HISTORY_SCOPE_MAIN).order_by("timestamp")

    qs = qs.values("timestamp", "power_kw", "irradiation", "air_temp", "pv_temp")
    df = pd.DataFrame.from_records(qs)
    if df.empty:
        print(f"[TRAIN] station {station.pk}: история пуста")
        return df

    df.rename(
        columns={
            "timestamp": "ds",
            "power_kw": "Power_KW",
            "irradiation": "Irradiation",
            "air_temp": "Air_Temp",
            "pv_temp": "PV_Temp",
        },
        inplace=True,
    )
    df["ds"] = pd.to_datetime(df["ds"])

    if history_station.pk != station.pk:
        scale_factor = _history_scale_factor(station, history_station)
        if scale_factor:
            df["Power_KW"] = df["Power_KW"] * scale_factor
            print(
                f"[TRAIN] station {station.pk}: история взята от {history_station.pk} "
                f"и масштабирована x{scale_factor:.3f}"
            )
        else:
            target_cap = _capacity_mw_from_fields(station)
            source_cap = _capacity_mw_from_fields(history_station)
            if target_cap and source_cap and source_cap > 0:
                auto_ratio = target_cap / source_cap
                if abs(auto_ratio - 1.0) >= 0.2:
                    df["Power_KW"] = df["Power_KW"] * auto_ratio
                    print(
                        f"[TRAIN] station {station.pk}: история от {history_station.pk} "
                        f"автомасштабирована по мощности x{auto_ratio:.3f}"
                    )
                else:
                    print(
                        f"[TRAIN] station {station.pk}: история взята от {history_station.pk} "
                        "без масштабирования"
                    )
            else:
                print(
                    f"[TRAIN] station {station.pk}: история взята от {history_station.pk} "
                    "без масштабирования"
                )

    numeric_cols = ["Power_KW", "Irradiation", "Air_Temp"]
    before = len(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=numeric_cols)
    removed = before - len(df)
    if removed:
        print(
            f"[TRAIN] station {station.pk}: удалено {removed} строк "
            f"с NaN/inf в {numeric_cols}"
        )

    return df


def compute_cap_mw(df: pd.DataFrame) -> float:
    """
    Грубая оценка установленной мощности станции (MW) по истории.
    """
    if df.empty:
        return 1.0
    max_kw = float(df["Power_KW"].max())
    # минимальный кап 0.5 МВт
    return max(0.5, max_kw / 1000.0)


def station_capacity_mw(station, df: pd.DataFrame) -> float:
    capacity_from_fields = _capacity_mw_from_fields(station)
    if capacity_from_fields:
        return capacity_from_fields

    return compute_cap_mw(df)


def _prepare_xgb_training_frame(df: pd.DataFrame, cap_mw: float) -> pd.DataFrame:
    """
    Подготовка выборки XGB:
    - обучаем на отношении к ожидаемой мощности (y_over_expected),
      чтобы убрать зависимость от масштаба станции и снизить риск "мелкого" XGB;
    - отсекаем явную ночную/нулевую часть.
    """
    out = df.copy()
    out["y_permw"] = (out["y"] / cap_mw).clip(lower=0)

    irr = pd.to_numeric(out.get("Irradiation"), errors="coerce").fillna(0.0)
    is_day_or_active = (irr > 20.0) | (out["y_permw"] > 0.02)
    out = out.loc[is_day_or_active].copy()

    expected_col = "tracker_calibrated_expected_mw" if "tracker_calibrated_expected_mw" in out.columns else "y_expected"
    y_expected = pd.to_numeric(out.get(expected_col), errors="coerce").fillna(0.0)
    expected_floor_mw = max(0.05 * float(cap_mw), 0.15)
    expected_base = np.maximum(y_expected, expected_floor_mw)
    out["y_over_expected"] = (out["y"] / np.maximum(expected_base, 1e-6)).clip(0.0, 1.6)

    return out.dropna(subset=["y_over_expected"])


def add_sun_geometry(df: pd.DataFrame, ds_col: str = "ds", lat_deg: float = 47.86) -> pd.DataFrame:
    lat = np.deg2rad(lat_deg)
    doy = df[ds_col].dt.dayofyear
    hour = df[ds_col].dt.hour
    hour_angle = np.deg2rad((hour - 12) * 15)
    decl = np.deg2rad(23.44) * np.sin(2 * np.pi * (284 + doy) / 365)
    sin_elev = (
        np.sin(lat) * np.sin(decl)
        + np.cos(lat) * np.cos(decl) * np.cos(hour_angle)
    )
    df["sun_elev_deg"] = np.rad2deg(np.arcsin(np.clip(sin_elev, -1, 1)))
    df["low_sun_flag"] = (df["sun_elev_deg"] < 15).astype(int)
    return df



def _is_single_axis_tracker(station) -> bool:
    mount_type = str(getattr(station, "mount_type", Station.MOUNT_FIXED) or Station.MOUNT_FIXED)
    mount_type = mount_type.strip().lower().replace("-", "_")
    return mount_type == Station.MOUNT_SINGLE_AXIS_TRACKER


def _tracker_profile_factor(df: pd.DataFrame) -> np.ndarray:
    hours = pd.to_datetime(df["ds"]).dt.hour.astype(int).to_numpy()
    irradiation = pd.to_numeric(df.get("Irradiation"), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    sun_elev = pd.to_numeric(df.get("sun_elev_deg", pd.Series(0.0, index=df.index)), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    daylight = (irradiation > 0.0) & (sun_elev > 0.0)
    factor = np.ones(len(df), dtype=float)

    shoulder_mask = daylight & (((hours >= 6) & (hours <= 10)) | ((hours >= 15) & (hours <= 19)))
    edge_strength = np.clip((45.0 - sun_elev) / 45.0, 0.0, 1.0)
    factor[shoulder_mask] *= 1.0 + 0.16 * edge_strength[shoulder_mask]

    midday_mask = daylight & (hours >= 11) & (hours <= 14)
    high_sun_strength = np.clip((sun_elev - 35.0) / 35.0, 0.0, 1.0)
    factor[midday_mask] *= 1.0 - 0.08 * high_sun_strength[midday_mask]
    return np.clip(factor, 0.0, 1.18)


def add_tracker_training_features(df: pd.DataFrame, station, cap_mw: float) -> pd.DataFrame:
    """Add tracker-specific expected curve columns used by retrained models."""
    out = df.copy()
    if not _is_single_axis_tracker(station):
        return out

    factor = _tracker_profile_factor(out)
    irradiation = pd.to_numeric(out.get("Irradiation"), errors="coerce").fillna(0.0)
    expected_physical = np.clip(cap_mw * (irradiation / 1000.0) * PR_FOR_EXPECTED * factor, 0.0, cap_mw)
    y = pd.to_numeric(out.get("y"), errors="coerce").fillna(0.0)

    denom = np.maximum(expected_physical, max(0.03 * float(cap_mw), 0.05))
    calib_raw = (y / denom).replace([np.inf, -np.inf], np.nan)
    valid = (irradiation >= 80.0) & calib_raw.notna() & (calib_raw > 0.2) & (calib_raw < 2.0)
    curve = calib_raw.loc[valid].groupby(out.loc[valid, "hour"]).median().clip(0.70, 1.30).to_dict()
    station_calibration = out["hour"].map(curve).fillna(1.0).astype(float)

    out["tracker_profile_factor"] = factor
    out["tracker_station_calibration_factor"] = station_calibration
    out["tracker_calibrated_expected_mw"] = np.clip(expected_physical * station_calibration, 0.0, cap_mw)
    out["tracker_calibrated_expected_log"] = np.log1p(out["tracker_calibrated_expected_mw"])
    print(
        f"[TRAIN] station {station.pk}: tracker features enabled, "
        f"calibration_hours={len(curve)}"
    )
    return out


def add_common_features(df: pd.DataFrame, cap_mw: float, ds_col: str = "ds") -> pd.DataFrame:
    """
    Добавляем все фичи, которые нужны и для XGB, и для NeuralProphet.

    На выходе есть:
      hour, month, hour_sin, month_sin,
      is_clear, morning_peak_boost, evening_penalty,
      overdrive_flag, midday_penalty, is_morning_active,
      sunrise_hour_flag, solar_ramp_factor, irradiation_x_ramp,
      y_expected_log
    """
    df = df.copy()
    df["ds"] = pd.to_datetime(df["ds"])

    df["hour"] = df["ds"].dt.hour
    df["month"] = df["ds"].dt.month

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df["is_clear"] = ((df["Irradiation"] > 200) & (df["Air_Temp"] > 0)).astype(int)

    # Мягкие утренние признаки: вместо агрессивного точечного буста
    # даём модели форму плавного разгона генерации в 06-08.
    df["sunrise_hour_flag"] = df["hour"].between(6, 8).astype(int)
    ramp_map = {6: 0.35, 7: 0.65, 8: 0.90}
    df["solar_ramp_factor"] = df["hour"].map(ramp_map).fillna(1.0).astype(float)
    df["irradiation_x_ramp"] = df["Irradiation"] * df["solar_ramp_factor"]

    # Старый morning_peak_boost оставляем для совместимости структуры признаков,
    # но больше не включаем его для 07:00/08:00 и не расширяем утро вверх.
    df["morning_peak_boost"] = ((df["hour"] == 6) & (df["Irradiation"] > 80)).astype(int)
    df["evening_penalty"] = ((df["hour"] == 19) & (df["Irradiation"] > 39)).astype(int)
    df["overdrive_flag"] = ((df["Irradiation"] > 950) & (df["Air_Temp"] > 30)).astype(int)
    df["midday_penalty"] = ((df["hour"] >= 12) & (df["hour"] <= 14)).astype(int)
    df["is_morning_active"] = ((df["hour"] == 6) & (df["Irradiation"] > 49)).astype(int)

    if "PV_Temp" not in df.columns:
        df["PV_Temp"] = np.nan
    if df["PV_Temp"].isna().mean() > 0.80:
        df["PV_Temp"] = df["Air_Temp"] + np.maximum(df["Irradiation"] - 50, 0) / 1000 * 20

    if "Wind_Speed" not in df.columns:
        df["Wind_Speed"] = np.nan

    df["y_expected"] = cap_mw * (df["Irradiation"] / 1000) * PR_FOR_EXPECTED
    df["y_expected"] = df["y_expected"].clip(upper=cap_mw * 0.95)
    df["y_expected_log"] = np.log1p(df["y_expected"] * 0.95)

    return df


def train_models_for_station(station) -> Tuple[int, Path | None, Path | None]:
    """
    Обучаем XGB(per-MW) и NeuralProphet(y) для одной станции.

    Сохраняем:
      - XGB:  <model_dir>/xgb_model.json + .meta.json
      - NP:   <model_dir>/np_model.np   + .meta.json

    Возвращаем: (кол-во строк истории, путь к NP, путь к XGB)
    """
    if not hasattr(station, "pk"):
        from stations.models import Station

        station = Station.objects.get(pk=station)

    df = get_history_dataframe(station)
    if df.empty:
        return 0, None, None

    cap_mw = station_capacity_mw(station, df)
    print(f"[TRAIN] station {station.pk}: оценка cap_mw={cap_mw:.3f}")

    lat_deg = getattr(station, "lat", None) or getattr(station, "latitude", None) or 47.86

    df["y"] = (df["Power_KW"] / 1000.0).clip(lower=0)
    df = df[df["y"] >= 0].copy()

    df = add_common_features(df, cap_mw, "ds")
    df = add_sun_geometry(df, "ds", float(lat_deg))
    df = add_tracker_training_features(df, station, cap_mw)
    n_rows = len(df)

    # =========================================================
    # 1) Обучение XGB per-MW
    # =========================================================
    X_cols = [
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
    tracker_expected_col = None
    if _is_single_axis_tracker(station) and "tracker_calibrated_expected_mw" in df.columns:
        tracker_expected_col = "tracker_calibrated_expected_mw"
        X_cols += [
            "tracker_profile_factor",
            "tracker_station_calibration_factor",
            "tracker_calibrated_expected_mw",
            "tracker_calibrated_expected_log",
        ]

    for col in X_cols:
        if col not in df.columns:
            df[col] = np.nan

    df_xgb = _prepare_xgb_training_frame(df, cap_mw=cap_mw)

    model_dir = _station_model_dir(station)
    xgb_path: Path | None = None
    if len(df_xgb) == 0:
        print(f"[TRAIN] station {station.pk}: нет строк для XGB после фильтрации")
    else:
        try:
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
            irr = pd.to_numeric(df_xgb.get("Irradiation"), errors="coerce").fillna(0.0)
            sample_weight = 1.0 + 3.0 * np.clip(irr / 800.0, 0.0, 1.0) + 2.0 * (df_xgb["y_over_expected"] > 0.80).astype(float)
            model_xgb.fit(df_xgb[X_cols], df_xgb["y_over_expected"], sample_weight=sample_weight)

            # Простая пост-калибровка масштаба XGB по train-набору,
            # чтобы избежать систематического занижения (microscopic XGB).
            pred_train = pd.to_numeric(model_xgb.predict(df_xgb[X_cols]), errors="coerce").astype(float)
            pred_train = np.clip(pred_train, 1e-4, None)
            ratio = pd.to_numeric(df_xgb["y_over_expected"], errors="coerce").astype(float) / pred_train
            ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
            if len(ratio) > 0:
                calib_mult = float(np.clip(np.median(ratio), 0.5, 8.0))
            else:
                calib_mult = 1.0

            xgb_path = model_dir / "xgb_model.json"
            model_xgb.save_model(str(xgb_path))

            meta_xgb = {
                "station_id": station.pk,
                "X_cols": X_cols,
                "cap_mw_used": cap_mw,
                "target": "y_over_expected = y / max(expected_col, floor)",
                "expected_col": tracker_expected_col or "y_expected",
                "y_expected_floor_mw": max(0.05 * float(cap_mw), 0.15),
                "train_filter": "(Irradiation > 20) or (y_permw > 0.02)",
                "weighted_fit": "1 + 3*clip(Irradiation/800,0,1) + 2*(y_over_expected>0.80)",
                "xgb_calib_mult": calib_mult,
                "train_rows_total": int(len(df)),
                "train_rows_used": int(len(df_xgb)),
                "xgb_version": getattr(xgb, "__version__", "unknown"),
            }
            meta_path = model_dir / "xgb_model.meta.json"
            meta_path.write_text(
                json.dumps(meta_xgb, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(
                f"[TRAIN] station {station.pk}: XGB(per-MW) обучен, "
                f"сохранён в {xgb_path}"
            )
        except Exception as e:
            import traceback

            print(f"[TRAIN] station {station.pk}: ОШИБКА при обучении XGB -> {e}")
            traceback.print_exc()
            xgb_path = None

    # =========================================================
    # 2) Обучение NeuralProphet на residual
    # =========================================================
    np_path: Path | None = None
    try:
        df_np = df.copy()
        np_base_expected_col = tracker_expected_col or "y_expected"
        df_np["y_residual"] = df_np["y"] - pd.to_numeric(df_np[np_base_expected_col], errors="coerce").fillna(0.0)

        print(
            f"[TRAIN] station {station.pk}: старт обучения NP(residual), строк={len(df_np)}"
        )

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

        reg_cols = [
            "Irradiation",
            "Air_Temp",
            "PV_Temp",
            "Wind_Speed",
            "hour_sin",
            "hour_cos",
            "month_sin",
            "month_cos",
            "is_clear",
            "y_expected_log",
            "morning_peak_boost",
            "evening_penalty",
            "overdrive_flag",
            "midday_penalty",
            "is_morning_active",
            "sun_elev_deg",
            "low_sun_flag",
            "sunrise_hour_flag",
            "solar_ramp_factor",
            "irradiation_x_ramp",
        ]
        if tracker_expected_col:
            reg_cols += [
                "tracker_profile_factor",
                "tracker_station_calibration_factor",
                "tracker_calibrated_expected_mw",
                "tracker_calibrated_expected_log",
            ]
        for col in reg_cols:
            m.add_future_regressor(col, normalize="minmax")

        df_fit = df_np[["ds", "y_residual"] + reg_cols].copy()
        df_fit.rename(columns={"y_residual": "y"}, inplace=True)

        fill_map: dict[str, float] = {}
        for col in reg_cols:
            if col not in df_fit.columns:
                df_fit[col] = np.nan
            med = df_fit[col].median(skipna=True)
            if pd.isna(med):
                med = 0.0
            fill_map[col] = float(med)

        df_fit = df_fit.dropna(subset=["ds", "y"]).copy()
        for col in reg_cols:
            df_fit[col] = pd.to_numeric(df_fit[col], errors="coerce").fillna(fill_map[col])

        if len(df_fit) < 500:
            raise RuntimeError(f"Too few rows for NP after cleaning: {len(df_fit)}")

        df_train = df_fit.copy()
        m.fit(df_train, freq="h")

        np_path = model_dir / "np_model.np"
        try:
            if hasattr(m, "save"):
                m.save(str(np_path))
            else:
                np_save(m, str(np_path))
            print(f"[TRAIN] station {station.pk}: NP(residual) сохранён в {np_path}")
        except Exception as e:
            print(f"[TRAIN] station {station.pk}: ОШИБКА сохранения NP(residual) -> {e}")
            raise

        features_reg = reg_cols
        np_meta_path = model_dir / "np_model.meta.json"
        np_meta = {
            "station_id": station.pk,
            "cap_mw": cap_mw,
            "pr_for_expected": PR_FOR_EXPECTED,
            "features_reg": features_reg,
            "fill_map": fill_map,
            "target": f"y_residual = y - {np_base_expected_col}",
            "base_expected_col": np_base_expected_col,
            "note": "NP residual. Missing regressor values are filled using train medians.",
            "np_version": getattr(neuralprophet, "__version__", "unknown"),
        }
        np_meta_path.write_text(
            json.dumps(np_meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[TRAIN] station {station.pk}: NP meta сохранён в {np_meta_path}")
    except Exception as e:
        import traceback

        print(f"[TRAIN] station {station.pk}: ОШИБКА при обучении NP(y) -> {e}")
        traceback.print_exc()
        np_path = None

    return n_rows, np_path, xgb_path
