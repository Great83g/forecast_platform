from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Any, Dict, List

import math

from django.apps import apps
from django.db import transaction
from django.utils import timezone


def _pick_model_field(Model, candidates: list[str]) -> str:
    """Return first existing field name from candidates."""
    fields = {f.name for f in Model._meta.get_fields()}
    for c in candidates:
        if c in fields:
            return c
    raise RuntimeError(
        f"No matching field in {Model.__name__}. "
        f"Candidates={candidates}, fields={sorted(fields)}"
    )


def _get_model_any(app_labels: list[str], model_names: list[str]):
    for app_label in app_labels:
        for name in model_names:
            try:
                m = apps.get_model(app_label, name)
                if m:
                    return m
            except Exception:
                continue
    return None


def _station_model():
    # максимально “живучий” поиск модели станции
    m = _get_model_any(
        app_labels=["solar", "stations"],
        model_names=["Station", "SolarStation", "PVStation", "SolarPlant", "StationsStation"],
    )
    if not m:
        raise RuntimeError(
            "Не нашёл модель станции. Проверь apps 'solar'/'stations' и имя модели станции."
        )
    return m


def _forecast_model():
    m = _get_model_any(
        app_labels=["solar", "stations"],
        model_names=["SolarForecast", "Forecast"],
    )
    if not m:
        raise RuntimeError(
            "Не нашёл модель прогноза (SolarForecast). Проверь app 'solar'/'stations'."
        )
    return m


def _get_station_capacity_kw(station: Any) -> float:
    cap_kw = getattr(station, "capacity_kw", None)
    if cap_kw is not None:
        try:
            v = float(cap_kw)
            if v > 0:
                return v
        except Exception:
            pass

    cap_mw = getattr(station, "capacity_mw", None)
    if cap_mw is not None:
        try:
            v = float(cap_mw)
            if v > 0:
                return v * 1000.0
        except Exception:
            pass

    # дефолт чтобы не падать
    return 10000.0  # 10 МВт


def _simple_heuristic_mw(irr: float, cap_kw: float) -> float:
    irr = max(0.0, float(irr or 0.0))
    kw = (irr / 1000.0) * cap_kw * 0.85
    mw = kw / 1000.0
    return max(0.0, mw)


@dataclass
class ForecastRow:
    ts: datetime
    irradiation: float
    air_temp: float
    wind_speed: float
    cloudcover: float
    humidity: float
    precip: float
    pred_np_kw: Optional[float]
    pred_xgb_kw: Optional[float]
    pred_heur_kw: float
    pred_final_kw: float


def train_models_for_station(station_id: int) -> Dict[str, Any]:
    """
    Заглушка обучения (чтобы кнопка "Обучить модели" могла работать позже).
    Реальное обучение NP/XGB подключим отдельным PR/патчем.
    """
    Station = _station_model()
    station = Station.objects.get(pk=station_id)
    return {"ok": True, "station": str(getattr(station, "name", station_id)), "trained": False}


@transaction.atomic
def run_forecast_for_station(station_id: int, days: int = 3) -> Dict[str, Any]:
    """
    Стабильный прогноз-минимум на N дней вперед по часу.
    Пишем в реальные поля SolarForecast:
      timestamp, pred_np, pred_xgb, pred_heur, pred_final, irradiation_fc, air_temp_fc, ...
    """
    Station = _station_model()
    Forecast = _forecast_model()

    station = Station.objects.get(pk=station_id)
    cap_kw = _get_station_capacity_kw(station)

    now = timezone.localtime()
    start = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    end = start + timedelta(days=int(days))

    # Найдём реальные имена полей в SolarForecast
    station_field = _pick_model_field(Forecast, ["station", "plant", "solar_station", "stations_station"])
    ts_field = _pick_model_field(Forecast, ["timestamp", "dt", "datetime", "date_time", "ds"])

    # поля прогнозов (в твоей модели они такие)
    pred_np_field = _pick_model_field(Forecast, ["pred_np"])
    pred_xgb_field = _pick_model_field(Forecast, ["pred_xgb"])
    pred_heur_field = _pick_model_field(Forecast, ["pred_heur"])
    pred_final_field = _pick_model_field(Forecast, ["pred_final"])

    # поля погоды (в твоей модели они такие)
    irr_fc_field = _pick_model_field(Forecast, ["irradiation_fc"])
    air_fc_field = _pick_model_field(Forecast, ["air_temp_fc"])
    wind_fc_field = _pick_model_field(Forecast, ["wind_speed_fc"])
    cloud_fc_field = _pick_model_field(Forecast, ["cloudcover_fc"])
    hum_fc_field = _pick_model_field(Forecast, ["humidity_fc"])
    precip_fc_field = _pick_model_field(Forecast, ["precip_fc"])

    # (created_at есть в твоей модели)
    created_field = None
    try:
        created_field = _pick_model_field(Forecast, ["created_at", "created", "created_on", "created_dt"])
    except Exception:
        created_field = None

    rows: List[ForecastRow] = []
    ts = start

    while ts < end:
        hour = ts.hour

        # простая “солнечная форма” (чтобы UI жил)
        daylight = 1.0 if 6 <= hour <= 18 else 0.0
        bell = math.exp(-((hour - 12) ** 2) / (2 * 3.0 ** 2))
        irradiation = 900.0 * daylight * bell  # W/m2

        # минимальная погода-заглушка
        air_temp = 10.0
        wind_speed = 3.0
        cloudcover = 30.0
        humidity = 50.0
        precip = 0.0

        heur_mw = _simple_heuristic_mw(irradiation, cap_kw)
        final_mw = heur_mw  # пока ансамбль = эвристика

        rows.append(
            ForecastRow(
                ts=ts,
                irradiation=irradiation,
                air_temp=air_temp,
                wind_speed=wind_speed,
                cloudcover=cloudcover,
                humidity=humidity,
                precip=precip,
                pred_np_kw=None,
                pred_xgb_kw=None,
                pred_heur_kw=float(heur_mw * 1000.0),   # MW -> kW
                pred_final_kw=float(final_mw * 1000.0), # MW -> kW
            )
        )
        ts += timedelta(hours=1)

    # чистим прогноз по станции и диапазону
    Forecast.objects.filter(**{station_field: station, f"{ts_field}__gte": start, f"{ts_field}__lt": end}).delete()

    objs = []
    now_ts = timezone.now()
    for r in rows:
        dt_value = r.ts
        if timezone.is_naive(dt_value):
            dt_value = timezone.make_aware(dt_value, timezone.get_default_timezone())

        kwargs = {
            station_field: station,
            ts_field: dt_value,

            pred_np_field: r.pred_np_kw,
            pred_xgb_field: r.pred_xgb_kw,
            pred_heur_field: r.pred_heur_kw,
            pred_final_field: r.pred_final_kw,

            irr_fc_field: float(r.irradiation),
            air_fc_field: float(r.air_temp),
            wind_fc_field: float(r.wind_speed),
            cloud_fc_field: float(r.cloudcover),
            hum_fc_field: float(r.humidity),
            precip_fc_field: float(r.precip),
        }
        if created_field:
            kwargs[created_field] = now_ts
        objs.append(Forecast(**kwargs))

    Forecast.objects.bulk_create(objs)
    return {"ok": True, "count": len(objs), "start": start.isoformat(), "end": end.isoformat()}

