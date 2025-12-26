# dashboard/views.py
from __future__ import annotations

from datetime import datetime
from io import BytesIO
from typing import Optional

import pandas as pd
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils import timezone
from django.conf import settings
from urllib.parse import urlencode

from stations.models import Station
from solar.models import SolarRecord, SolarForecast

from .forms import StationForm, UploadHistoryForm

# forecast service (обязательно должен быть)
from .services.forecast_engine import run_forecast_for_station

# train service (может быть/не быть — не валим портал)
try:
    from .services.train_models import train_models_for_station
except Exception:
    train_models_for_station = None


# ----------------------------
# helpers
# ----------------------------
def _parse_date(s: str) -> Optional[datetime]:
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass
    return None


def _excel_safe_datetime(series: pd.Series) -> pd.Series:
    """
    Excel не поддерживает tz-aware datetime.
    Приводим к naive.
    """
    s = pd.to_datetime(series, errors="coerce")
    try:
        # если tz-aware -> убираем tz
        if getattr(s.dt, "tz", None) is not None:
            s = s.dt.tz_convert(timezone.get_current_timezone())
            s = s.dt.tz_localize(None)
        else:
            s = s.dt.tz_localize(None)
    except Exception:
        # если уже naive — ок
        pass
    return s


def _localize_timestamp(value):
    if value is None or pd.isna(value):
        return value
    try:
        if timezone.is_naive(value):
            return timezone.make_aware(value, timezone.get_current_timezone())
        return timezone.localtime(value)
    except Exception:
        return value


# ----------------------------
# stations
# ----------------------------
@login_required
def station_list(request):
    stations = Station.objects.all().order_by("id")
    return render(request, "dashboard/station_list.html", {"stations": stations})


@login_required
def station_create(request):
    if request.method == "POST":
        form = StationForm(request.POST)
        if form.is_valid():
            st = form.save()
            messages.success(request, "Станция создана.")
            return redirect("dashboard:station-detail", pk=st.pk)
        messages.error(request, "Ошибка в форме станции.")
    else:
        form = StationForm()

    return render(request, "dashboard/station_create.html", {"form": form})


@login_required
def station_edit(request, pk: int):
    st = get_object_or_404(Station, pk=pk)

    if request.method == "POST":
        form = StationForm(request.POST, instance=st)
        if form.is_valid():
            form.save()
            messages.success(request, "Станция обновлена.")
            return redirect("dashboard:station-detail", pk=st.pk)
        messages.error(request, "Ошибка в форме станции.")
    else:
        form = StationForm(instance=st)

    return render(request, "dashboard/station_edit.html", {"station": st, "form": form})


@login_required
def station_detail(request, pk: int):
    st = get_object_or_404(Station, pk=pk)
    return render(request, "dashboard/station_detail.html", {"station": st})


# ----------------------------
# history upload/export
# ----------------------------
@login_required
def station_upload(request, pk: int):
    st = get_object_or_404(Station, pk=pk)

    if request.method == "POST":
        if request.POST.get("action") == "clear":
            SolarRecord.objects.filter(station=st).delete()
            messages.success(request, "История очищена.")
            return redirect("dashboard:station-upload", pk=pk)

        form = UploadHistoryForm(request.POST, request.FILES)
        if not form.is_valid():
            messages.error(request, "Ошибка формы загрузки.")
            return redirect("dashboard:station-upload", pk=pk)

        f = request.FILES.get("file")
        if not f:
            messages.error(request, "Файл не выбран.")
            return redirect("dashboard:station-upload", pk=pk)

        try:
            if f.name.lower().endswith(".csv"):
                df = pd.read_csv(f)
            else:
                df = pd.read_excel(f)
        except Exception as e:
            messages.error(request, f"Не удалось прочитать файл: {e}")
            return redirect("dashboard:station-upload", pk=pk)

        # нормализуем названия колонок: убираем пробелы и приводим к нижнему регистру
        df.columns = [str(c).strip().lower() for c in df.columns]

        # поддержим разные названия колонок
        col_ts = "timestamp" if "timestamp" in df.columns else ("ds" if "ds" in df.columns else None)
        col_y = "power_kw" if "power_kw" in df.columns else ("y" if "y" in df.columns else None)

        if not col_ts or not col_y:
            messages.error(request, "Нужны колонки timestamp/ds и power_kw/y (регистр не важен).")
            return redirect("dashboard:station-upload", pk=pk)

        df[col_ts] = pd.to_datetime(df[col_ts], errors="coerce")
        df[col_y] = pd.to_numeric(df[col_y], errors="coerce")

        # опциональные колонки
        for c in ["irradiation", "air_temp", "pv_temp"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna(subset=[col_ts]).sort_values(col_ts).reset_index(drop=True)

        # полностью заменяем историю
        SolarRecord.objects.filter(station=st).delete()

        objs = []
        for _, r in df.iterrows():
            objs.append(
                SolarRecord(
                    station=st,
                    timestamp=r[col_ts].to_pydatetime(),
                    power_kw=float(r[col_y]) if pd.notna(r[col_y]) else None,
                    irradiation=float(r["irradiation"]) if "irradiation" in df.columns and pd.notna(r.get("irradiation")) else None,
                    air_temp=float(r["air_temp"]) if "air_temp" in df.columns and pd.notna(r.get("air_temp")) else None,
                    pv_temp=float(r["pv_temp"]) if "pv_temp" in df.columns and pd.notna(r.get("pv_temp")) else None,
                )
            )

        SolarRecord.objects.bulk_create(objs, batch_size=1000)
        messages.success(request, f"История загружена: {len(objs)} строк.")
        return redirect("dashboard:station-upload", pk=pk)

    # GET + показ истории
    form = UploadHistoryForm()
    from_s = request.GET.get("from") or ""
    to_s = request.GET.get("to") or ""
    dt_from = _parse_date(from_s)
    dt_to = _parse_date(to_s)

    qs = SolarRecord.objects.filter(station=st).order_by("timestamp")
    total_count = qs.count()
    if dt_from:
        qs = qs.filter(timestamp__gte=dt_from)
    if dt_to:
        qs = qs.filter(timestamp__lte=dt_to)
    history = list(qs)

    return render(
        request,
        "dashboard/station_upload.html",
        {
            "station": st,
            "form": form,
            "history": history,
            "from_date": from_s,
            "to_date": to_s,
            "total_count": total_count,
            "history_count": len(history),
        },
    )


@login_required
def station_export_history(request, pk: int):
    st = get_object_or_404(Station, pk=pk)

    qs = SolarRecord.objects.filter(station=st).order_by("timestamp")
    data = list(qs.values("timestamp", "power_kw", "irradiation", "air_temp", "pv_temp"))
    df = pd.DataFrame(data)

    if not df.empty and "timestamp" in df.columns:
        df["timestamp"] = _excel_safe_datetime(df["timestamp"])

    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="history")
    out.seek(0)

    resp = HttpResponse(
        out.getvalue(),
        content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    resp["Content-Disposition"] = f'attachment; filename="history_station_{st.pk}.xlsx"'
    return resp


# ----------------------------
# training (обязательно)
# ----------------------------
@login_required
def station_train(request, pk: int):
    """
    Страница обучения (GET) + запуск обучения (POST).
    """
    st = get_object_or_404(Station, pk=pk)

    if request.method == "POST":
        if train_models_for_station is None:
            messages.error(request, "train_models_for_station не найден. Проверь dashboard/services/train_models.py")
            return redirect("dashboard:station-train", pk=pk)

        try:
            res = train_models_for_station(st)
            # res может быть dict/str — покажем как есть
            messages.success(request, f"Обучение запущено/выполнено: {res}")
        except Exception as e:
            messages.error(request, f"Ошибка обучения: {e}")

        return redirect("dashboard:station-detail", pk=pk)

    # GET
    # покажем статус: есть ли модели в models_cache (если хочешь — добавим позже красиво)
    return render(request, "dashboard/station_train.html", {"station": st})


@login_required
def station_train_models(request, pk: int):
    """
    Совместимость с url: /train-models/ (у тебя в urls он указывает на station_train)
    """
    return station_train(request, pk=pk)


# ----------------------------
# forecast list/run/export/clear
# ----------------------------
@login_required
def station_forecast_list(request, pk: int):
    st = get_object_or_404(Station, pk=pk)

    days = int(request.GET.get("days", "1") or 1)
    selected_providers = request.GET.getlist("providers") or getattr(
        settings,
        "FORECAST_WEATHER_PROVIDERS",
        ["visual_crossing"],
    )
    from_s = request.GET.get("from") or ""
    to_s = request.GET.get("to") or ""
    dt_from = _parse_date(from_s)
    dt_to = _parse_date(to_s)

    qs = SolarForecast.objects.filter(station=st).order_by("timestamp")
    if dt_from:
        qs = qs.filter(timestamp__gte=dt_from)
    if dt_to:
        qs = qs.filter(timestamp__lte=dt_to)

    forecasts_raw = list(qs)

    forecasts = [
        {
            "timestamp": _localize_timestamp(f.timestamp),
            "pred_final_kw": f.pred_final,
            "pred_np_kw": f.pred_np,
            "pred_xgb_kw": f.pred_xgb,
            "pred_heur_kw": f.pred_heur,
        }
        for f in forecasts_raw
    ]

    return render(
        request,
        "dashboard/station_forecast_list.html",
        {
            "station": st,
            "forecasts": forecasts,
            "days": days,
            "selected_providers": selected_providers,
            "from": from_s,
            "to": to_s,
            "count": len(forecasts),
        },
    )


@login_required
def station_forecast_run(request, pk: int):
    st = get_object_or_404(Station, pk=pk)
    days = int(request.GET.get("days", "1") or 1)
    providers = request.GET.getlist("providers") or None

    try:
        res = run_forecast_for_station(st.pk, days=days, providers=providers)
        if res.get("ok"):
            msg = f"Прогноз построен: {res.get('count')} строк, days={days}, weather={res.get('weather_source')}"
            if not res.get("np_ok"):
                np_err = res.get("np_error") or "FAIL"
                msg += f" | NP: {np_err}"
            if not res.get("xgb_ok"):
                xgb_err = res.get("xgb_error") or "FAIL"
                msg += f" | XGB: {xgb_err}"
            messages.success(request, msg)
        else:
            messages.error(request, f"Ошибка прогноза: {res}")
    except Exception as e:
        messages.error(request, f"Ошибка запуска прогноза: {e}")

    query = urlencode({"days": days, "providers": providers or []}, doseq=True)
    return redirect(f"{reverse('dashboard:station-forecast-list', kwargs={'pk': st.pk})}?{query}")


@login_required
def station_forecast_clear(request, pk: int):
    st = get_object_or_404(Station, pk=pk)
    SolarForecast.objects.filter(station=st).delete()
    messages.success(request, "Прогноз очищен.")
    return redirect("dashboard:station-forecast-list", pk=st.pk)


@login_required
def station_forecast_export(request, pk: int):
    st = get_object_or_404(Station, pk=pk)

    from_s = request.GET.get("from") or ""
    to_s = request.GET.get("to") or ""
    dt_from = _parse_date(from_s)
    dt_to = _parse_date(to_s)

    qs = SolarForecast.objects.filter(station=st).order_by("timestamp")
    if dt_from:
        qs = qs.filter(timestamp__gte=dt_from)
    if dt_to:
        qs = qs.filter(timestamp__lte=dt_to)

    data = list(
        qs.values(
            "timestamp",
            "pred_np",
            "pred_xgb",
            "pred_heur",
            "irradiation_fc",
            "air_temp_fc",
            "wind_speed_fc",
            "cloudcover_fc",
            "humidity_fc",
            "precip_fc",
            "pred_final",
        )
    )
    df = pd.DataFrame(data)

    if df.empty:
        df = pd.DataFrame(
            columns=[
                "timestamp",
                "pred_np",
                "pred_xgb",
                "pred_heur",
                "irradiation_fc",
                "air_temp_fc",
                "wind_speed_fc",
                "cloudcover_fc",
                "humidity_fc",
                "precip_fc",
                "pred_final",
            ]
        )

    if "timestamp" in df.columns and not df.empty:
        ts = df["timestamp"].apply(_localize_timestamp)
        df["timestamp"] = _excel_safe_datetime(ts)

    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="forecast")
    out.seek(0)

    resp = HttpResponse(
        out.getvalue(),
        content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    resp["Content-Disposition"] = f'attachment; filename="forecast_station_{st.pk}.xlsx"'
    return resp
