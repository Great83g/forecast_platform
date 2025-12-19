from __future__ import annotations

from datetime import datetime
from io import BytesIO

import pandas as pd

from django.contrib import messages


def _pick_model_field(Model, candidates):
    fields = {f.name for f in Model._meta.get_fields()}
    for c in candidates:
        if c in fields:
            return c
    raise RuntimeError(f"No matching field in {Model.__name__}; candidates={candidates}; fields={sorted(fields)}")

from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone

from django.apps import apps

from .services.run_forecast import run_forecast_for_station, train_models_for_station


def _get_model(app_label: str, candidates: list[str]):
    for name in candidates:
        try:
            m = apps.get_model(app_label, name)
            if m:
                return m
        except Exception:
            continue
    return None


def StationModel():
    m = _get_model("solar", ["SolarStation", "Station", "PVStation"]) or _get_model("stations", ["Station"])
    if not m:
        raise RuntimeError("Не нашёл модель станции. Проверь solar/stations models.py.")
    return m


def RecordModel():
    m = _get_model("solar", ["SolarRecord", "Record"]) or _get_model("stations", ["SolarRecord", "Record"])
    if not m:
        raise RuntimeError("Не нашёл модель истории (SolarRecord).")
    return m


def ForecastModel():
    m = _get_model("solar", ["SolarForecast", "Forecast"]) or _get_model("stations", ["SolarForecast", "Forecast"])
    if not m:
        raise RuntimeError("Не нашёл модель прогноза (SolarForecast).")
    return m


@login_required
def station_list(request):
    Station = StationModel()
    stations = Station.objects.all().order_by("id")
    return render(request, "dashboard/station_list.html", {"stations": stations})


@login_required
def station_create(request):
    Station = StationModel()
    if request.method == "POST":
        name = request.POST.get("name", "").strip()
        operator = request.POST.get("operator", "").strip()
        if not name:
            messages.error(request, "Имя станции обязательно.")
        else:
            st = Station.objects.create(name=name, operator=operator)
            messages.success(request, "Станция создана.")
            return redirect("dashboard:station-detail", pk=st.pk)
    return render(request, "dashboard/station_create.html")


@login_required
def station_edit(request, pk: int):
    Station = StationModel()
    st = get_object_or_404(Station, pk=pk)

    if request.method == "POST":
        st.name = request.POST.get("name", st.name).strip()
        st.operator = request.POST.get("operator", getattr(st, "operator", "")).strip()
        # опционально
        for f in ["capacity_kw", "capacity_mw", "lat", "lon"]:
            if hasattr(st, f) and f in request.POST:
                val = request.POST.get(f, "").strip()
                if val != "":
                    try:
                        setattr(st, f, float(val))
                    except Exception:
                        pass
        st.save()
        messages.success(request, "Станция обновлена.")
        return redirect("dashboard:station-detail", pk=st.pk)

    return render(request, "dashboard/station_edit.html", {"station": st})


@login_required
def station_detail(request, pk: int):
    Station = StationModel()
    st = get_object_or_404(Station, pk=pk)
    return render(request, "dashboard/station_detail.html", {"station": st})


@login_required
def station_upload(request, pk: int):
    Station = StationModel()
    Record = RecordModel()
    st = get_object_or_404(Station, pk=pk)

    # фильтр дат
    from_date = request.GET.get("from", "")
    to_date = request.GET.get("to", "")

    qs = Record.objects.filter(station=st).order_by("-timestamp")
    if from_date:
        qs = qs.filter(timestamp__date__gte=from_date)
    if to_date:
        qs = qs.filter(timestamp__date__lte=to_date)

    if request.method == "POST":
        action = request.POST.get("action", "")

        if action == "clear":
            Record.objects.filter(station=st).delete()
            messages.success(request, "История очищена.")
            return redirect("dashboard:station-upload", pk=pk)

        if action == "upload":
            f = request.FILES.get("file")
            if not f:
                messages.error(request, "Файл не выбран.")
                return redirect("dashboard:station-upload", pk=pk)

            # CSV/XLSX
            if f.name.lower().endswith(".csv"):
                df = pd.read_csv(f)
            else:
                df = pd.read_excel(f)

            # ожидаемые колонки
            # ds, Irradiation, Air_Temp, PV_Temp, Power_KW
            colmap = {c.lower(): c for c in df.columns}
            def pick(name): return colmap.get(name.lower())

            ds_col = pick("ds")
            irr_col = pick("Irradiation")
            air_col = pick("Air_Temp")
            pv_col = pick("PV_Temp")
            pwr_col = pick("Power_KW")

            if not ds_col or not irr_col or not air_col or not pv_col or not pwr_col:
                messages.error(request, "Не вижу нужные колонки: ds, Irradiation, Air_Temp, PV_Temp, Power_KW")
                return redirect("dashboard:station-upload", pk=pk)

            df = df[[ds_col, irr_col, air_col, pv_col, pwr_col]].copy()
            df.columns = ["ds", "irradiation", "air_temp", "pv_temp", "power_kw"]
            df["ds"] = pd.to_datetime(df["ds"], errors="coerce")
            df = df.dropna(subset=["ds"])

            objs = []
            for r in df.itertuples(index=False):
                objs.append(
                    Record(
                        station=st,
                        timestamp=timezone.make_aware(r.ds) if timezone.is_naive(r.ds) else r.ds,
                        irradiation=float(r.irradiation or 0),
                        air_temp=float(r.air_temp or 0),
                        pv_temp=float(r.pv_temp or 0),
                        power_kw=float(r.power_kw or 0),
                    )
                )
            Record.objects.bulk_create(objs)
            messages.success(request, f"Загружено записей: {len(objs)}")
            return redirect("dashboard:station-upload", pk=pk)

    ctx = {
        "station": st,
        "history": qs[:2000],  # чтобы страницу не убивать
        "total_count": Record.objects.filter(station=st).count(),
        "history_count": qs.count(),
        "from_date": from_date,
        "to_date": to_date,
        "days_options": list(range(1, 8)),
        "days_selected": int(request.GET.get("days", 3)) if str(request.GET.get("days", "")).isdigit() else 3,
    }
    return render(request, "dashboard/station_upload.html", ctx)


@login_required
def station_export_history(request, pk: int):
    Station = StationModel()
    Record = RecordModel()
    st = get_object_or_404(Station, pk=pk)

    from_date = request.GET.get("from", "")
    to_date = request.GET.get("to", "")

    qs = Record.objects.filter(station=st).order_by("timestamp")
    if from_date:
        qs = qs.filter(timestamp__date__gte=from_date)
    if to_date:
        qs = qs.filter(timestamp__date__lte=to_date)

    rows = list(qs.values("timestamp", "irradiation", "air_temp", "pv_temp", "power_kw"))
    df = pd.DataFrame(rows)
    if not df.empty:
        df.rename(columns={ts_field: "ds"}, inplace=True)

    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="history")

    resp = HttpResponse(bio.getvalue(), content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    resp["Content-Disposition"] = f'attachment; filename="history_station_{pk}.xlsx"'
    return resp


@login_required
def station_train(request, pk: int):
    st = get_object_or_404(StationModel(), pk=pk)
    res = train_models_for_station(st.pk)
    if res.get("ok"):
        messages.success(request, "Обучение выполнено (пока заглушка, но кнопка рабочая).")
    else:
        messages.error(request, "Ошибка обучения.")
    return redirect("dashboard:station-detail", pk=pk)


@login_required
def station_forecast_list(request, pk: int):
    Station = StationModel()
    Forecast = ForecastModel()
    st = get_object_or_404(Station, pk=pk)

    from_date = request.GET.get("from", "")
    to_date = request.GET.get("to", "")

    ts_field = _pick_model_field(Forecast, ["timestamp", "dt", "datetime", "date_time", "ds"])

    qs = Forecast.objects.filter(station=st).order_by(ts_field)
    if from_date:
        qs = qs.filter(**{f"{ts_field}__date__gte": from_date})
    if to_date:
        qs = qs.filter(**{f"{ts_field}__date__lte": to_date})

    ctx = {
        "station": st,
        "rows": qs[:5000],
        "total_count": Forecast.objects.filter(station=st).count(),
        "filtered_count": qs.count(),
        "from_date": from_date,
        "to_date": to_date,
        "days_options": list(range(1, 8)),
        "days_selected": int(request.GET.get("days", 3)) if str(request.GET.get("days", "")).isdigit() else 3,
    }
    return render(request, "dashboard/station_forecast.html", ctx)


@login_required
def station_forecast_run(request, pk: int):
    st = get_object_or_404(StationModel(), pk=pk)

    # days can be passed as ?days=1..7
    try:
        days = int(request.GET.get("days", 3))
    except (TypeError, ValueError):
        days = 3
    days = max(1, min(7, days))

    # IMPORTANT: call service with keyword arg to avoid positional mismatch
    run_forecast_for_station(st.pk, days=days)
    messages.success(request, f"Прогноз построен на {days} дн.")
    return redirect("dashboard:station-forecast-list", pk=pk)


@login_required
def station_forecast_export(request, pk: int):
    Station = StationModel()
    Forecast = ForecastModel()
    st = get_object_or_404(Station, pk=pk)

    from_date = request.GET.get("from", "")
    to_date = request.GET.get("to", "")

    ts_field = _pick_model_field(Forecast, ["timestamp", "dt", "datetime", "date_time", "ds"])

    qs = Forecast.objects.filter(station=st).order_by(ts_field)
    if from_date:
        qs = qs.filter(**{f"{ts_field}__date__gte": from_date})
    if to_date:
        qs = qs.filter(**{f"{ts_field}__date__lte": to_date})

    fields = [ts_field, "np_mw", "xgb_mw", "heuristic_mw", "ensemble_mw"]
    rows = list(qs.values(*fields))
    df = pd.DataFrame(rows)
    if not df.empty:
        df.rename(columns={ts_field: "ds"}, inplace=True)

    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="forecast")

    resp = HttpResponse(bio.getvalue(), content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    resp["Content-Disposition"] = f'attachment; filename="forecast_station_{pk}.xlsx"'
    return resp


@login_required
def station_forecast_clear(request, pk: int):
    Station = StationModel()
    Forecast = ForecastModel()
    st = get_object_or_404(Station, pk=pk)
    Forecast.objects.filter(station=st).delete()
    messages.success(request, "Прогноз очищен.")
    return redirect("dashboard:station-forecast-list", pk=pk)

