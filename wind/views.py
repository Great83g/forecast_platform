from __future__ import annotations

from datetime import datetime
from io import BytesIO
from typing import Optional

import pandas as pd
from django.contrib import messages
from django.core.files.base import ContentFile
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils import timezone

from dashboard.forms import UploadHistoryForm
from dashboard.models import ForecastSchedule
from dashboard.services.open_meteo import fetch_open_meteo_hourly
from dashboard.services.vc_weather import fetch_visual_crossing_hourly
from dashboard.services.forecast_reports import send_report_email
from stations.models import Organization, OrganizationMember, Station

from .forms import WindForecastScheduleForm, WindStationForm, WindStationProfileForm
from .models import WindForecast, WindRecord


def _wind_station_queryset_for_user(user):
    org_ids = Organization.objects.filter(owner=user).values_list("id", flat=True)
    member_org_ids = OrganizationMember.objects.filter(user=user).values_list("organization_id", flat=True)
    return Station.objects.filter(org_id__in=(org_ids.union(member_org_ids)), station_kind=Station.KIND_WIND).distinct()


def _get_wind_station_or_404(user, pk: int) -> Station:
    return get_object_or_404(_wind_station_queryset_for_user(user), pk=pk)


def _parse_date(s: str) -> Optional[datetime]:
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass
    return None


def _aware_datetime(value: Optional[datetime], *, end_of_day: bool = False) -> Optional[datetime]:
    if value is None:
        return None
    if end_of_day:
        value = value.replace(hour=23, minute=59, second=59, microsecond=999999)
    if timezone.is_naive(value):
        return timezone.make_aware(value, timezone.get_current_timezone())
    return timezone.localtime(value, timezone.get_current_timezone())


def _normalize_history_scope(value: str) -> str:
    return "test" if value == "test" else "main"


def _normalize_forecast_scope(value: str) -> str:
    return "test" if value == "test" else "main"




def _excel_safe_datetime(series: pd.Series) -> pd.Series:
    s = pd.to_datetime(series, errors="coerce")
    try:
        if getattr(s.dt, "tz", None) is not None:
            s = s.dt.tz_convert(timezone.get_current_timezone())
            s = s.dt.tz_localize(None)
        else:
            s = s.dt.tz_localize(None)
    except Exception:
        pass
    return s

def _parse_history_datetime(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", dayfirst=True)
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(series[missing], errors="coerce")
    return parsed


def _wind_power_kw_for_speed(station: Station, speed_ms: float | None) -> float:
    if speed_ms is None or pd.isna(speed_ms):
        return 0.0

    profile = getattr(station, "wind_profile", None)
    cut_in = float(getattr(profile, "cut_in_speed_ms", 3.0) or 3.0)
    rated = float(getattr(profile, "rated_speed_ms", 12.0) or 12.0)
    cut_out = float(getattr(profile, "cut_out_speed_ms", 25.0) or 25.0)
    capacity_kw = float(getattr(station, "capacity_ac_kw", 0.0) or 0.0)

    v = float(speed_ms)
    if v < cut_in or v >= cut_out or capacity_kw <= 0:
        return 0.0
    if v >= rated:
        return capacity_kw
    denominator = max(rated - cut_in, 0.1)
    normalized = max((v - cut_in) / denominator, 0.0)
    return capacity_kw * (normalized ** 3)




def _normalize_recipients(value: str) -> list[str]:
    if not value:
        return []
    return [p.strip() for p in value.replace(";", ",").split(",") if p.strip()]


def _build_wind_forecast_report(station: Station, scope: str, days: int, weather_source: str, recipients_raw: str = ""):
    from dashboard.models import ForecastReport

    qs = WindForecast.objects.filter(station=station, forecast_scope=scope).order_by("timestamp")
    data = list(
        qs.values(
            "timestamp",
            "pred_heur",
            "pred_final",
            "wind_speed_fc",
            "air_temp_fc",
            "cloudcover_fc",
            "humidity_fc",
            "precip_fc",
            "weather_source",
        )
    )
    df = pd.DataFrame(data)
    if not df.empty and "timestamp" in df.columns:
        df["timestamp"] = _excel_safe_datetime(df["timestamp"])
        for col in ["pred_heur", "pred_final"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce") / 1000.0
                df.rename(columns={col: f"{col}_mw"}, inplace=True)

    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="wind_forecast")
    out.seek(0)

    stamp = timezone.localtime(timezone.now()).strftime("%Y%m%d_%H%M%S")
    filename = f"wind_forecast_station_{station.pk}_{stamp}.xlsx"
    report = ForecastReport(
        station=station,
        days=days,
        weather_source=weather_source or "",
        recipients=", ".join(_normalize_recipients(recipients_raw)),
    )
    report.file.save(filename, ContentFile(out.getvalue()), save=False)
    report.save()
    return report

def _fetch_weather_for_wind(station: Station, days: int, providers: list[str]) -> tuple[pd.DataFrame, str, list[str]]:
    errors = []
    results = []
    used_sources = []

    for provider in providers:
        if provider == "visual_crossing":
            wr = fetch_visual_crossing_hourly(station.latitude, station.longitude, days)
        elif provider == "open_meteo":
            wr = fetch_open_meteo_hourly(station.latitude, station.longitude, days, tz_name=station.timezone)
        else:
            continue

        if wr.ok and not wr.df.empty:
            df = wr.df.copy()
            if "wind_direction" in df.columns and "wind_direction_fc" not in df.columns:
                df["wind_direction_fc"] = pd.to_numeric(df.get("wind_direction"), errors="coerce")
            results.append(df)
            used_sources.append(wr.source)
        else:
            errors.append(f"{provider}: {wr.error or 'empty'}")

    if not results:
        return pd.DataFrame(), ",".join(used_sources), errors

    merged = pd.concat(results, ignore_index=True)
    numeric_cols = [
        c for c in ["irradiation", "air_temp", "wind_speed", "wind_direction_fc", "cloudcover", "humidity", "precip"] if c in merged.columns
    ]
    merged = merged.groupby("ds", as_index=False)[numeric_cols].mean(numeric_only=True)
    merged = merged.sort_values("ds").reset_index(drop=True)
    return merged, "+".join(sorted(set(used_sources))), errors


@login_required
def station_list(request):
    stations = list(_wind_station_queryset_for_user(request.user).select_related("org", "wind_profile").order_by("sort_order", "id"))
    return render(request, "wind/station_list.html", {"stations": stations})


@login_required
def station_create(request):
    if request.method == "POST":
        station_form = WindStationForm(request.POST, user=request.user)
        profile_form = WindStationProfileForm(request.POST)
        if station_form.is_valid() and profile_form.is_valid():
            st = station_form.save(commit=False)
            st.station_kind = Station.KIND_WIND

            profile = profile_form.save(commit=False)
            installed_capacity_kw = profile.installed_capacity_kw
            st.capacity_ac_kw = installed_capacity_kw
            st.capacity_dc_kw = installed_capacity_kw
            st.capacity_mw = installed_capacity_kw / 1000.0

            if st.data_shift_hours in (None, ""):
                st.data_shift_hours = 0

            st.save()
            profile.station = st
            profile.save()

            messages.success(request, "Ветростанция создана.")
            return redirect("wind:station-list")
        messages.error(request, "Проверьте форму: есть ошибки в параметрах ветростанции.")
    else:
        station_form = WindStationForm(user=request.user)
        profile_form = WindStationProfileForm()

    return render(
        request,
        "wind/station_create.html",
        {
            "station_form": station_form,
            "profile_form": profile_form,
        },
    )


@login_required
def station_detail(request, pk: int):
    station = _get_wind_station_or_404(request.user, pk)
    return render(request, "wind/station_detail.html", {"station": station})


@login_required
def station_upload(request, pk: int):
    station = _get_wind_station_or_404(request.user, pk)
    history_scope = _normalize_history_scope(request.POST.get("history_scope") or request.GET.get("history_scope") or "main")

    if request.method == "POST":
        if request.POST.get("action") == "clear":
            from_s = (request.POST.get("from") or "").strip()
            to_s = (request.POST.get("to") or "").strip()
            dt_from = _aware_datetime(_parse_date(from_s), end_of_day=False)
            dt_to = _aware_datetime(_parse_date(to_s), end_of_day=True)

            clear_qs = WindRecord.objects.filter(station=station, history_scope=history_scope)
            if dt_from:
                clear_qs = clear_qs.filter(timestamp__gte=dt_from)
            if dt_to:
                clear_qs = clear_qs.filter(timestamp__lte=dt_to)

            deleted_count, _ = clear_qs.delete()
            if dt_from or dt_to:
                messages.success(request, f"Удалено записей истории: {deleted_count} (по фильтру дат).")
            else:
                messages.success(request, f"История очищена: удалено {deleted_count} записей.")
            return redirect(f"{reverse('wind:station-upload', kwargs={'pk': pk})}?history_scope={history_scope}")

        form = UploadHistoryForm(request.POST, request.FILES)
        if not form.is_valid():
            messages.error(request, "Ошибка формы загрузки.")
            return redirect(f"{reverse('wind:station-upload', kwargs={'pk': pk})}?history_scope={history_scope}")

        f = request.FILES.get("file")
        if not f:
            messages.error(request, "Файл не выбран.")
            return redirect(f"{reverse('wind:station-upload', kwargs={'pk': pk})}?history_scope={history_scope}")

        try:
            if f.name.lower().endswith(".csv"):
                df = pd.read_csv(f)
            else:
                df = pd.read_excel(f)
        except Exception as e:
            messages.error(request, f"Не удалось прочитать файл: {e}")
            return redirect(f"{reverse('wind:station-upload', kwargs={'pk': pk})}?history_scope={history_scope}")

        df.columns = [str(c).strip().lower() for c in df.columns]

        col_ts = "timestamp" if "timestamp" in df.columns else ("ds" if "ds" in df.columns else None)
        col_power = "power_kw" if "power_kw" in df.columns else ("y" if "y" in df.columns else None)
        col_wind_speed = "wind_speed_ms" if "wind_speed_ms" in df.columns else (
            "wind_speed" if "wind_speed" in df.columns else ("ws" if "ws" in df.columns else None)
        )

        if not col_ts or not col_power:
            messages.error(request, "Нужны колонки timestamp/ds и power_kw/y (регистр не важен).")
            return redirect(f"{reverse('wind:station-upload', kwargs={'pk': pk})}?history_scope={history_scope}")

        df[col_ts] = _parse_history_datetime(df[col_ts])
        df[col_power] = pd.to_numeric(df[col_power], errors="coerce")

        if col_wind_speed:
            df[col_wind_speed] = pd.to_numeric(df[col_wind_speed], errors="coerce")
        for c in ["wind_direction_deg", "air_temp", "air_density"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna(subset=[col_ts]).sort_values(col_ts).reset_index(drop=True)

        WindRecord.objects.filter(station=station, history_scope=history_scope).delete()

        objs = []
        for _, r in df.iterrows():
            objs.append(
                WindRecord(
                    station=station,
                    history_scope=history_scope,
                    timestamp=r[col_ts].to_pydatetime(),
                    power_kw=float(r[col_power]) if pd.notna(r[col_power]) else None,
                    wind_speed_ms=float(r[col_wind_speed]) if col_wind_speed and pd.notna(r.get(col_wind_speed)) else None,
                    wind_direction_deg=float(r["wind_direction_deg"]) if "wind_direction_deg" in df.columns and pd.notna(r.get("wind_direction_deg")) else None,
                    air_temp=float(r["air_temp"]) if "air_temp" in df.columns and pd.notna(r.get("air_temp")) else None,
                    air_density=float(r["air_density"]) if "air_density" in df.columns and pd.notna(r.get("air_density")) else None,
                )
            )

        WindRecord.objects.bulk_create(objs, batch_size=1000)
        messages.success(request, f"История ветра загружена: {len(objs)} строк.")
        return redirect(f"{reverse('wind:station-upload', kwargs={'pk': pk})}?history_scope={history_scope}")

    form = UploadHistoryForm()
    from_s = request.GET.get("from") or ""
    to_s = request.GET.get("to") or ""
    dt_from = _parse_date(from_s)
    dt_to = _parse_date(to_s)

    qs = WindRecord.objects.filter(station=station, history_scope=history_scope).order_by("timestamp")
    total_count = qs.count()
    if dt_from:
        qs = qs.filter(timestamp__gte=dt_from)
    if dt_to:
        qs = qs.filter(timestamp__lte=dt_to)
    history = list(qs)

    return render(
        request,
        "wind/station_upload.html",
        {
            "station": station,
            "form": form,
            "history": history,
            "from_date": from_s,
            "to_date": to_s,
            "total_count": total_count,
            "history_count": len(history),
            "history_scope": history_scope,
        },
    )


@login_required
def station_export_history(request, pk: int):
    station = _get_wind_station_or_404(request.user, pk)
    history_scope = _normalize_history_scope(request.GET.get("history_scope") or "main")
    qs = WindRecord.objects.filter(station=station, history_scope=history_scope).order_by("timestamp")
    data = list(qs.values("timestamp", "power_kw", "wind_speed_ms", "wind_direction_deg", "air_temp", "air_density"))
    df = pd.DataFrame(data)

    if not df.empty and "timestamp" in df.columns:
        df["timestamp"] = _excel_safe_datetime(df["timestamp"])

    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="wind_history")
    out.seek(0)

    resp = HttpResponse(
        out.getvalue(),
        content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    resp["Content-Disposition"] = f'attachment; filename="wind_history_station_{station.pk}.xlsx"'
    return resp


@login_required
def station_forecast_list(request, pk: int):
    station = _get_wind_station_or_404(request.user, pk)
    scope = _normalize_forecast_scope(request.GET.get("scope") or "test")

    from_s = request.GET.get("from") or ""
    to_s = request.GET.get("to") or ""
    dt_from = _parse_date(from_s)
    dt_to = _parse_date(to_s)

    qs = WindForecast.objects.filter(station=station, forecast_scope=scope).order_by("timestamp")
    if dt_from:
        qs = qs.filter(timestamp__date__gte=dt_from.date())
    if dt_to:
        qs = qs.filter(timestamp__date__lte=dt_to.date())
    forecasts = list(qs)

    schedule = ForecastSchedule.objects.filter(station=station).first()
    initial = {
        "enabled": bool(schedule.enabled) if schedule else False,
        "run_time": schedule.run_time if schedule else datetime.strptime("06:00", "%H:%M").time(),
        "days": schedule.days if schedule else 2,
        "providers": [p.strip() for p in (schedule.providers or "visual_crossing,open_meteo").split(",") if p.strip()] if schedule else ["visual_crossing", "open_meteo"],
        "emails": schedule.emails if schedule else "",
        "auto_send": False,
    }
    schedule_form = WindForecastScheduleForm(initial=initial)

    return render(
        request,
        "wind/station_forecast_list.html",
        {
            "station": station,
            "forecasts": forecasts,
            "scope": scope,
            "from": from_s,
            "to": to_s,
            "schedule_form": schedule_form,
            "count": len(forecasts),
            "emails": initial.get("emails", ""),
        },
    )


@login_required
def station_forecast_run(request, pk: int):
    station = _get_wind_station_or_404(request.user, pk)
    days = int(request.GET.get("days") or 2)
    scope = _normalize_forecast_scope(request.GET.get("scope") or "test")
    providers = request.GET.getlist("providers") or ["visual_crossing", "open_meteo"]
    emails_raw = request.GET.get("emails") or ""
    auto_send = request.GET.get("auto_send") in {"1", "true", "on", "yes"}

    if station.latitude is None or station.longitude is None:
        messages.error(request, "Для прогноза задайте координаты станции.")
        return redirect(f"{reverse('wind:station-forecast-list', kwargs={'pk': station.pk})}?scope={scope}")

    weather_df, weather_source, errors = _fetch_weather_for_wind(station, days, providers)
    if weather_df.empty:
        messages.error(request, f"Не удалось получить погоду: {'; '.join(errors) or 'empty response'}")
        return redirect(f"{reverse('wind:station-forecast-list', kwargs={'pk': station.pk})}?scope={scope}")

    WindForecast.objects.filter(station=station, forecast_scope=scope).delete()

    objs = []
    for _, row in weather_df.iterrows():
        speed = pd.to_numeric(row.get("wind_speed"), errors="coerce")
        pred = _wind_power_kw_for_speed(station, speed)
        objs.append(
            WindForecast(
                station=station,
                forecast_scope=scope,
                timestamp=row.get("ds").to_pydatetime() if hasattr(row.get("ds"), "to_pydatetime") else row.get("ds"),
                pred_heur=pred,
                pred_final=pred,
                weather_source=weather_source,
                air_temp_fc=float(row.get("air_temp")) if pd.notna(row.get("air_temp")) else None,
                wind_speed_fc=float(speed) if pd.notna(speed) else None,
                wind_direction_fc=float(row.get("wind_direction_fc")) if pd.notna(row.get("wind_direction_fc")) else None,
                cloudcover_fc=float(row.get("cloudcover")) if pd.notna(row.get("cloudcover")) else None,
                humidity_fc=float(row.get("humidity")) if pd.notna(row.get("humidity")) else None,
                precip_fc=float(row.get("precip")) if pd.notna(row.get("precip")) else None,
            )
        )

    WindForecast.objects.bulk_create(objs, batch_size=1000)

    msg = f"Ветровой прогноз построен: {len(objs)} строк, source={weather_source}, scope={scope}"
    try:
        report = _build_wind_forecast_report(station, scope=scope, days=days, weather_source=weather_source, recipients_raw=emails_raw)
        msg += f". Отчёт: {report.file.name}"
        if auto_send and emails_raw:
            sent = send_report_email(report, _normalize_recipients(emails_raw), station.name, days)
            msg += " | Email: отправлен" if sent else " | Email: ошибка отправки"
        elif emails_raw:
            msg += " | Email: авто-отправка выключена"
    except Exception as report_error:
        msg += f" | Отчёт/Email: ошибка ({report_error})"

    messages.success(request, msg)
    return redirect(f"{reverse('wind:station-forecast-list', kwargs={'pk': station.pk})}?scope={scope}")


@login_required
def station_forecast_schedule_update(request, pk: int):
    station = _get_wind_station_or_404(request.user, pk)
    if request.method != "POST":
        return redirect("wind:station-forecast-list", pk=station.pk)

    form = WindForecastScheduleForm(request.POST)
    if not form.is_valid():
        messages.error(request, "Ошибка в настройках автопрогноза ветра.")
        return redirect("wind:station-forecast-list", pk=station.pk)

    schedule, _ = ForecastSchedule.objects.get_or_create(station=station)
    schedule.enabled = form.cleaned_data["enabled"]
    schedule.run_time = form.cleaned_data["run_time"]
    schedule.days = form.cleaned_data["days"]
    schedule.providers = ",".join(form.cleaned_data.get("providers") or [])
    schedule.emails = form.cleaned_data.get("emails", "")
    schedule.save()

    messages.success(request, "Настройки автопрогноза ветра сохранены.")
    return redirect(f"{reverse('wind:station-forecast-list', kwargs={'pk': station.pk})}?scope=main")


@login_required
def station_forecast_clear(request, pk: int):
    station = _get_wind_station_or_404(request.user, pk)
    scope = _normalize_forecast_scope(request.POST.get("scope") or request.GET.get("scope") or "main")
    deleted, _ = WindForecast.objects.filter(station=station, forecast_scope=scope).delete()
    messages.success(request, f"Прогноз очищен: удалено {deleted} строк (scope={scope}).")
    return redirect(f"{reverse('wind:station-forecast-list', kwargs={'pk': station.pk})}?scope={scope}")


@login_required
def station_forecast_export(request, pk: int):
    station = _get_wind_station_or_404(request.user, pk)
    scope = _normalize_forecast_scope(request.GET.get("scope") or "main")

    qs = WindForecast.objects.filter(station=station, forecast_scope=scope).order_by("timestamp")
    data = list(
        qs.values(
            "timestamp",
            "pred_heur",
            "pred_final",
            "weather_source",
            "air_temp_fc",
            "wind_speed_fc",
            "wind_direction_fc",
            "cloudcover_fc",
            "humidity_fc",
            "precip_fc",
        )
    )
    df = pd.DataFrame(data)

    if not df.empty and "timestamp" in df.columns:
        df["timestamp"] = _excel_safe_datetime(df["timestamp"])

    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="wind_forecast")
    out.seek(0)

    resp = HttpResponse(
        out.getvalue(),
        content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    resp["Content-Disposition"] = f'attachment; filename="wind_forecast_station_{station.pk}.xlsx"'
    return resp


@login_required
def station_train(request, pk: int):
    station = _get_wind_station_or_404(request.user, pk)
    return render(request, "wind/station_train.html", {"station": station})
