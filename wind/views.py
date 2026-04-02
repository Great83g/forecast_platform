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

from dashboard.forms import UploadHistoryForm
from stations.models import Organization, OrganizationMember, Station

from .forms import WindStationForm, WindStationProfileForm
from .models import WindRecord


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


def _parse_history_datetime(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", dayfirst=True)
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(series[missing], errors="coerce")
    return parsed


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
    return render(request, "wind/station_forecast_list.html", {"station": station})


@login_required
def station_train(request, pk: int):
    station = _get_wind_station_or_404(request.user, pk)
    return render(request, "wind/station_train.html", {"station": station})
