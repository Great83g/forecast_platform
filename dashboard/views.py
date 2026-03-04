# dashboard/views.py
from __future__ import annotations

from collections import defaultdict
import importlib
import logging
from datetime import datetime, timedelta
from io import BytesIO
from typing import Optional

import pandas as pd
from django.db.models import Avg
from django.db.models.functions import TruncHour
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils import timezone
from django.conf import settings
from urllib.parse import urlencode

from stations.models import Organization, OrganizationMember, Station
from solar.models import SolarRecord, SolarForecast

from .forms import StationForm, UploadHistoryForm, ForecastEmailForm, ForecastScheduleForm

# forecast service (обязательно должен быть)
from .services.forecast_engine import run_forecast_for_station
from .services.forecast_reports import build_forecast_report, send_report_email
from .services.forecast_scheduler import run_scheduled_forecasts
from .services.history_autofill import run_auto_history_updates
from .models import ForecastSchedule

logger = logging.getLogger(__name__)

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




def _aware_datetime(value: Optional[datetime], *, end_of_day: bool = False) -> Optional[datetime]:
    if value is None:
        return None
    if end_of_day:
        value = value.replace(hour=23, minute=59, second=59, microsecond=999999)
    if timezone.is_naive(value):
        return timezone.make_aware(value, timezone.get_current_timezone())
    return timezone.localtime(value, timezone.get_current_timezone())

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


def _parse_history_datetime(series: pd.Series) -> pd.Series:
    """
    Нормализует даты из истории с приоритетом day-first форматов.

    Это защищает от неверной интерпретации дат вида 01/03/2026
    (как 3 января вместо 1 марта) при загрузке CSV/XLSX.
    """
    parsed = pd.to_datetime(series, errors="coerce", dayfirst=True)

    # fallback для форматов, которые лучше читаются без dayfirst
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(series[missing], errors="coerce")

    return parsed


def _normalize_forecast_scope(value: str) -> str:
    if value == "test":
        return "test"
    return "main"


def _normalize_history_scope(value: str) -> str:
    if value == "test":
        return "test"
    return "main"


def _localize_timestamp(value):
    if value is None or pd.isna(value):
        return value
    try:
        if timezone.is_naive(value):
            return timezone.make_aware(value, timezone.get_current_timezone())
        return timezone.localtime(value)
    except Exception:
        return value

def _station_queryset_for_user(user):
    if user.is_superuser:
        return Station.objects.all()
    return Station.objects.filter(org__memberships__user=user).distinct()


def _get_station_or_404(user, pk: int):
    return get_object_or_404(_station_queryset_for_user(user), pk=pk)





def _station_write_denied_message(station):
    org = station.org
    if hasattr(org, "write_access_reason"):
        return org.write_access_reason()
    return "Запись данных временно недоступна."


def _ensure_station_write_access(request, station):
    org = station.org
    if hasattr(org, "can_write") and org.can_write():
        return True
    messages.error(request, _station_write_denied_message(station))
    return False

# ----------------------------
# stations
# ----------------------------
@login_required
def station_list(request):
    stations = _station_queryset_for_user(request.user).select_related("org").order_by("sort_order", "id")
    org_memberships = OrganizationMember.objects.filter(user=request.user).select_related("organization")
    onboarding_items = [
        "Добавьте первую станцию",
        "Загрузите исторические данные",
        "Запустите прогноз и проверьте отчёт",
        "Пригласите коллег в организацию",
    ]
    blocked_orgs = [m.organization for m in org_memberships if hasattr(m.organization, "can_write") and not m.organization.can_write()]
    return render(
        request,
        "dashboard/station_list.html",
        {"stations": stations, "onboarding_items": onboarding_items, "blocked_orgs": blocked_orgs},
    )


@login_required
def station_move(request, pk: int, direction: str):
    if request.method != "POST":
        return redirect("dashboard:station-list")

    st = _get_station_or_404(request.user, pk)

    if direction not in {"up", "down"}:
        messages.error(request, "Неизвестное направление перемещения.")
        return redirect("dashboard:station-list")

    siblings = list(
        _station_queryset_for_user(request.user)
        .filter(org=st.org)
        .order_by("sort_order", "id")
        .only("id", "sort_order")
    )
    current_idx = next((i for i, item in enumerate(siblings) if item.id == st.id), None)
    if current_idx is None:
        return redirect("dashboard:station-list")

    target_idx = current_idx - 1 if direction == "up" else current_idx + 1
    if target_idx < 0 or target_idx >= len(siblings):
        return redirect("dashboard:station-list")

    target = siblings[target_idx]
    Station.objects.filter(id=st.id).update(sort_order=target.sort_order)
    Station.objects.filter(id=target.id).update(sort_order=st.sort_order)

    return redirect("dashboard:station-list")

@login_required
def station_create(request):
    if request.method == "POST":
        form = StationForm(request.POST, user=request.user)
        if form.is_valid():
            org = form.cleaned_data.get("org")
            if org is not None and hasattr(org, "can_write") and not org.can_write():
                messages.error(request, org.write_access_reason())
                return render(request, "dashboard/station_create.html", {"form": form})
            st = form.save()
            messages.success(request, "Станция создана.")
            return redirect("dashboard:station-detail", pk=st.pk)
        messages.error(request, "Ошибка в форме станции.")
    else:
        form = StationForm(user=request.user)

    return render(request, "dashboard/station_create.html", {"form": form})




def _run_station_auto_history_fill_safe(station: Station) -> int:
    if not getattr(station, "auto_history_enabled", False):
        return 0

    try:
        module = importlib.import_module("dashboard.services.history_autofill")
        upsert = getattr(module, "upsert_station_history_from_share")
        return int(upsert(station) or 0)
    except Exception:
        logger.exception("Auto history fill on station save failed station_id=%s", station.pk)
        return 0


@login_required
def station_edit(request, pk: int):
    st = _get_station_or_404(request.user, pk)
    if not _ensure_station_write_access(request, st):
        return redirect("dashboard:station-detail", pk=st.pk)

    if request.method == "POST":
        form = StationForm(request.POST, instance=st, user=request.user)
        if form.is_valid():
            if not _ensure_station_write_access(request, st):
                return redirect("dashboard:station-detail", pk=st.pk)
            st = form.save()
            folder_ready = st.ensure_import_folder()
            details = getattr(st, "_last_import_folder_error", "")
            if not folder_ready:
                details_text = f" Детали: {details}" if details else ""
                messages.warning(
                    request,
                    f"Не удалось проверить/создать папку автоимпорта: {st.auto_history_folder}. "
                    "Проверьте права на /mnt/share для пользователя сервиса."
                    f"{details_text}",
                )
            elif (st.auto_history_folder or "").startswith("/tmp/forecast_platform_auto_history"):
                details_text = f" Причина: {details}" if details else ""
                messages.warning(
                    request,
                    f"Папка на /mnt/share недоступна, станция переключена на fallback: {st.auto_history_folder}."
                    f"{details_text}",
                )

            imported_rows = _run_station_auto_history_fill_safe(st)
            if st.auto_history_enabled:
                if imported_rows > 0:
                    messages.success(request, f"Автоистория обновлена: {imported_rows} строк.")
                else:
                    messages.warning(request, "Автоистория включена, но новых строк не найдено. Проверьте /mnt/share и имена файлов.")
            messages.success(request, "Станция обновлена.")
            return redirect("dashboard:station-detail", pk=st.pk)
        messages.error(request, "Ошибка в форме станции.")
    else:
        st.ensure_import_folder()
        st.refresh_from_db(fields=["auto_history_folder"])
        form = StationForm(instance=st, user=request.user)

    return render(request, "dashboard/station_edit.html", {"station": st, "form": form})


@login_required
def station_detail(request, pk: int):
    st = _get_station_or_404(request.user, pk)

    date_from = request.GET.get("date_from") or ""
    date_to = request.GET.get("date_to") or ""

    if not date_from and not date_to:
        now_local = timezone.localtime()
        default_from = (now_local - timedelta(days=30)).date()
        default_to = now_local.date()
        date_from = default_from.isoformat()
        date_to = default_to.isoformat()

    dt_from = _aware_datetime(_parse_date(date_from))
    dt_to = _aware_datetime(_parse_date(date_to), end_of_day=True)

    history_qs = SolarRecord.objects.filter(station=st, history_scope=SolarRecord.HISTORY_SCOPE_MAIN)
    forecast_qs = SolarForecast.objects.filter(station=st, forecast_scope=SolarForecast.SCOPE_MAIN)

    if dt_from:
        history_qs = history_qs.filter(timestamp__gte=dt_from)
        forecast_qs = forecast_qs.filter(timestamp__gte=dt_from)
    if dt_to:
        history_qs = history_qs.filter(timestamp__lte=dt_to)
        forecast_qs = forecast_qs.filter(timestamp__lte=dt_to)

    is_single_day_range = bool(dt_from and dt_to and dt_from.date() == dt_to.date())

    if is_single_day_range:
        history_rows = (
            history_qs.annotate(bucket=TruncHour("timestamp"))
            .values("bucket")
            .annotate(
                power_kw=Avg("power_kw"),
                irradiation=Avg("irradiation"),
                air_temp=Avg("air_temp"),
            )
            .order_by("bucket")
        )
        forecast_rows = (
            forecast_qs.annotate(bucket=TruncHour("timestamp"))
            .values("bucket")
            .annotate(
                pred_final=Avg("pred_final"),
                irradiation_fc=Avg("irradiation_fc"),
                air_temp_fc=Avg("air_temp_fc"),
            )
            .order_by("bucket")
        )
        history_map = {
            row["bucket"]: float(row["power_kw"])
            for row in history_rows
            if row.get("power_kw") is not None
        }
        forecast_map = {
            row["bucket"]: float(row["pred_final"])
            for row in forecast_rows
            if row.get("pred_final") is not None
        }
        irr_fact_map = {
            row["bucket"]: float(row["irradiation"])
            for row in history_rows
            if row.get("irradiation") is not None
        }
        irr_plan_map = {
            row["bucket"]: float(row["irradiation_fc"])
            for row in forecast_rows
            if row.get("irradiation_fc") is not None
        }
        temp_fact_map = {
            row["bucket"]: float(row["air_temp"])
            for row in history_rows
            if row.get("air_temp") is not None
        }
        temp_plan_map = {
            row["bucket"]: float(row["air_temp_fc"])
            for row in forecast_rows
            if row.get("air_temp_fc") is not None
        }
    else:
        history_rows = history_qs.values("timestamp").annotate(
            power_kw=Avg("power_kw"),
            irradiation=Avg("irradiation"),
            air_temp=Avg("air_temp"),
        ).order_by("timestamp")
        forecast_rows = forecast_qs.values("timestamp").annotate(
            pred_final=Avg("pred_final"),
            irradiation_fc=Avg("irradiation_fc"),
            air_temp_fc=Avg("air_temp_fc"),
        ).order_by("timestamp")
        history_map = {
            row["timestamp"]: float(row["power_kw"])
            for row in history_rows
            if row.get("power_kw") is not None
        }
        forecast_map = {
            row["timestamp"]: float(row["pred_final"])
            for row in forecast_rows
            if row.get("pred_final") is not None
        }
        irr_fact_map = {
            row["timestamp"]: float(row["irradiation"])
            for row in history_rows
            if row.get("irradiation") is not None
        }
        irr_plan_map = {
            row["timestamp"]: float(row["irradiation_fc"])
            for row in forecast_rows
            if row.get("irradiation_fc") is not None
        }
        temp_fact_map = {
            row["timestamp"]: float(row["air_temp"])
            for row in history_rows
            if row.get("air_temp") is not None
        }
        temp_plan_map = {
            row["timestamp"]: float(row["air_temp_fc"])
            for row in forecast_rows
            if row.get("air_temp_fc") is not None
        }

    merged_points = []
    labels = []
    fact_series = []
    plan_series = []
    irr_fact_series = []
    irr_plan_series = []
    temp_fact_series = []
    temp_plan_series = []
    fact_energy_kwh = 0.0
    plan_energy_kwh = 0.0
    mape_values = []
    mape_points_count = 0
    all_timestamps = sorted(
        set(history_map.keys())
        | set(forecast_map.keys())
        | set(irr_fact_map.keys())
        | set(irr_plan_map.keys())
        | set(temp_fact_map.keys())
        | set(temp_plan_map.keys())
    )
    for ts in all_timestamps:
        fact_kw = history_map.get(ts)
        plan_kw = forecast_map.get(ts)
        merged_points.append(
            {
                "timestamp": ts,
                "fact_mw": (fact_kw / 1000.0) if fact_kw is not None else None,
                "plan_mw": (plan_kw / 1000.0) if plan_kw is not None else None,
            }
        )
        ts_local = timezone.localtime(ts) if timezone.is_aware(ts) else ts
        labels.append(ts_local.strftime("%H:%M") if is_single_day_range else ts_local.strftime("%d.%m %H:%M"))
        fact_series.append(round(fact_kw / 1000.0, 4) if fact_kw is not None else None)
        plan_series.append(round(plan_kw / 1000.0, 4) if plan_kw is not None else None)
        irr_fact_series.append(round(irr_fact_map.get(ts), 2) if irr_fact_map.get(ts) is not None else None)
        irr_plan_series.append(round(irr_plan_map.get(ts), 2) if irr_plan_map.get(ts) is not None else None)
        temp_fact_series.append(round(temp_fact_map.get(ts), 2) if temp_fact_map.get(ts) is not None else None)
        temp_plan_series.append(round(temp_plan_map.get(ts), 2) if temp_plan_map.get(ts) is not None else None)

        # Суммируем мощность по шагам ряда как приближение энергии (кВт·ч).
        # Для дневного почасового разреза это близко к фактической суточной энергии,
        # для произвольного диапазона даёт агрегированный итог за период.
        if fact_kw is not None:
            fact_energy_kwh += fact_kw
        if plan_kw is not None:
            plan_energy_kwh += plan_kw


    deviation_kwh = fact_energy_kwh - plan_energy_kwh
    deviation_percent = (deviation_kwh / plan_energy_kwh * 100.0) if plan_energy_kwh else None

    fact_values = [value for value in history_map.values() if value is not None]
    peak_fact_kw = max(fact_values) if fact_values else 0.0
    min_fact_for_mape_kw = max(1.0, peak_fact_kw * 0.10)

    for ts in all_timestamps:
        fact_kw = history_map.get(ts)
        plan_kw = forecast_map.get(ts)
        if fact_kw is None or plan_kw is None or fact_kw <= 0:
            continue
        if fact_kw < min_fact_for_mape_kw:
            continue
        mape_values.append(abs((fact_kw - plan_kw) / fact_kw) * 100.0)

    mape_points_count = len(mape_values)
    if mape_values:
        mape_percent = sum(mape_values) / len(mape_values)
    else:
        mape_percent = None

    context = {
        "station": st,
        "date_from": date_from,
        "date_to": date_to,
        "labels": labels,
        "fact_series": fact_series,
        "plan_series": plan_series,
        "irr_fact_series": irr_fact_series,
        "irr_plan_series": irr_plan_series,
        "temp_fact_series": temp_fact_series,
        "temp_plan_series": temp_plan_series,
        "points_count": len(merged_points),
        "comparison_rows": merged_points[:200],
        "is_single_day_range": is_single_day_range,
        "fact_energy_kwh": round(fact_energy_kwh),
        "plan_energy_kwh": round(plan_energy_kwh),
        "deviation_kwh": round(deviation_kwh),
        "deviation_percent": round(deviation_percent, 1) if deviation_percent is not None else None,
        "mape_percent": round(mape_percent, 1) if mape_percent is not None else None,
        "mape_points_count": mape_points_count,
        "export_query": urlencode({"date_from": date_from, "date_to": date_to}),
    }
    return render(request, "dashboard/station_detail.html", context)


@login_required
def station_plan_fact_export(request, pk: int):
    st = _get_station_or_404(request.user, pk)

    date_from = request.GET.get("date_from") or ""
    date_to = request.GET.get("date_to") or ""
    dt_from = _aware_datetime(_parse_date(date_from))
    dt_to = _aware_datetime(_parse_date(date_to), end_of_day=True)

    history_qs = SolarRecord.objects.filter(station=st, history_scope=SolarRecord.HISTORY_SCOPE_MAIN)
    forecast_qs = SolarForecast.objects.filter(station=st, forecast_scope=SolarForecast.SCOPE_MAIN)
    if dt_from:
        history_qs = history_qs.filter(timestamp__gte=dt_from)
        forecast_qs = forecast_qs.filter(timestamp__gte=dt_from)
    if dt_to:
        history_qs = history_qs.filter(timestamp__lte=dt_to)
        forecast_qs = forecast_qs.filter(timestamp__lte=dt_to)

    history_df = pd.DataFrame(list(history_qs.values("timestamp").annotate(fact_kw=Avg("power_kw")).order_by("timestamp")))
    forecast_df = pd.DataFrame(list(forecast_qs.values("timestamp").annotate(plan_kw=Avg("pred_final")).order_by("timestamp")))

    if history_df.empty:
        history_df = pd.DataFrame(columns=["timestamp", "fact_kw"])
    if forecast_df.empty:
        forecast_df = pd.DataFrame(columns=["timestamp", "plan_kw"])

    df = history_df.merge(forecast_df, on="timestamp", how="outer").sort_values("timestamp")
    if not df.empty:
        df["timestamp"] = _excel_safe_datetime(df["timestamp"])

    fact_kw = pd.to_numeric(df.get("fact_kw"), errors="coerce")
    plan_kw = pd.to_numeric(df.get("plan_kw"), errors="coerce")
    df["fact_mw"] = (fact_kw / 1000.0).round(4)
    df["plan_mw"] = (plan_kw / 1000.0).round(4)

    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="plan_fact")
    out.seek(0)

    response = HttpResponse(
        out.getvalue(),
        content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    response["Content-Disposition"] = f'attachment; filename="plan_fact_station_{st.pk}.xlsx"'
    return response


# ----------------------------
# history upload/export
# ----------------------------
@login_required
def station_upload(request, pk: int):
    st = _get_station_or_404(request.user, pk)
    if not _ensure_station_write_access(request, st):
        return redirect("dashboard:station-detail", pk=st.pk)
    history_scope = _normalize_history_scope(request.POST.get("history_scope") or request.GET.get("history_scope") or "main")

    if request.method == "POST":
        if request.POST.get("action") == "clear":
            SolarRecord.objects.filter(station=st, history_scope=history_scope).delete()
            messages.success(request, "История очищена.")
            return redirect(f"{reverse('dashboard:station-upload', kwargs={'pk': pk})}?history_scope={history_scope}")

        form = UploadHistoryForm(request.POST, request.FILES)
        if not form.is_valid():
            messages.error(request, "Ошибка формы загрузки.")
            return redirect(f"{reverse('dashboard:station-upload', kwargs={'pk': pk})}?history_scope={history_scope}")

        f = request.FILES.get("file")
        if not f:
            messages.error(request, "Файл не выбран.")
            return redirect(f"{reverse('dashboard:station-upload', kwargs={'pk': pk})}?history_scope={history_scope}")

        try:
            if f.name.lower().endswith(".csv"):
                df = pd.read_csv(f)
            else:
                df = pd.read_excel(f)
        except Exception as e:
            messages.error(request, f"Не удалось прочитать файл: {e}")
            return redirect(f"{reverse('dashboard:station-upload', kwargs={'pk': pk})}?history_scope={history_scope}")

        # нормализуем названия колонок: убираем пробелы и приводим к нижнему регистру
        df.columns = [str(c).strip().lower() for c in df.columns]

        # поддержим разные названия колонок
        col_ts = "timestamp" if "timestamp" in df.columns else ("ds" if "ds" in df.columns else None)
        col_y = "power_kw" if "power_kw" in df.columns else ("y" if "y" in df.columns else None)

        if not col_ts or not col_y:
            messages.error(request, "Нужны колонки timestamp/ds и power_kw/y (регистр не важен).")
            return redirect(f"{reverse('dashboard:station-upload', kwargs={'pk': pk})}?history_scope={history_scope}")

        df[col_ts] = _parse_history_datetime(df[col_ts])
        df[col_y] = pd.to_numeric(df[col_y], errors="coerce")

        # опциональные колонки
        for c in ["irradiation", "air_temp", "pv_temp"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        df = df.dropna(subset=[col_ts]).sort_values(col_ts).reset_index(drop=True)

        # полностью заменяем историю
        SolarRecord.objects.filter(station=st, history_scope=history_scope).delete()

        objs = []
        for _, r in df.iterrows():
            objs.append(
                SolarRecord(
                    station=st,
                    history_scope=history_scope,
                    timestamp=r[col_ts].to_pydatetime(),
                    power_kw=float(r[col_y]) if pd.notna(r[col_y]) else None,
                    irradiation=float(r["irradiation"]) if "irradiation" in df.columns and pd.notna(r.get("irradiation")) else None,
                    air_temp=float(r["air_temp"]) if "air_temp" in df.columns and pd.notna(r.get("air_temp")) else None,
                    pv_temp=float(r["pv_temp"]) if "pv_temp" in df.columns and pd.notna(r.get("pv_temp")) else None,
                )
            )

        SolarRecord.objects.bulk_create(objs, batch_size=1000)
        messages.success(request, f"История загружена: {len(objs)} строк.")
        return redirect(f"{reverse('dashboard:station-upload', kwargs={'pk': pk})}?history_scope={history_scope}")

    # GET + показ истории
    form = UploadHistoryForm()
    from_s = request.GET.get("from") or ""
    to_s = request.GET.get("to") or ""
    dt_from = _parse_date(from_s)
    dt_to = _parse_date(to_s)

    qs = SolarRecord.objects.filter(station=st, history_scope=history_scope).order_by("timestamp")
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
            "history_scope": history_scope,
        },
    )


@login_required
def station_export_history(request, pk: int):
    st = _get_station_or_404(request.user, pk)

    history_scope = _normalize_history_scope(request.GET.get("history_scope") or "main")
    qs = SolarRecord.objects.filter(station=st, history_scope=history_scope).order_by("timestamp")
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
    st = _get_station_or_404(request.user, pk)
    if not _ensure_station_write_access(request, st):
        return redirect("dashboard:station-detail", pk=st.pk)

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
    st = _get_station_or_404(request.user, pk)

    days = int(request.GET.get("days", "7") or 7)
    open_meteo_only = request.GET.get("open_meteo_only") in {"1", "true", "on", "yes"}
    horizon_mode = request.GET.get("horizon_mode") or ""
    selected_providers = request.GET.getlist("providers") or getattr(
        settings,
        "FORECAST_WEATHER_PROVIDERS",
        ["visual_crossing"],
    )
    email_form = ForecastEmailForm(initial={"emails": request.GET.get("emails", "")})
    manual_snow_enable = request.GET.get("manual_snow_enable") in {"1", "true", "on", "yes"}
    manual_snow_factor = request.GET.get("manual_snow_factor") or ""
    manual_snow_dates = request.GET.get("manual_snow_dates") or ""
    target_dates_raw = request.GET.get("target_dates") or ""
    manual_auto_send = request.GET.get("manual_auto_send") in {"1", "true", "on", "yes"}
    forecast_scope = _normalize_forecast_scope(request.GET.get("scope") or "test")
    schedule = ForecastSchedule.objects.filter(station=st).first()
    if schedule:
        if not manual_snow_enable and request.GET.get("manual_snow_enable") is None:
            manual_snow_enable = schedule.manual_snow_enable
        if manual_snow_factor == "" and schedule.manual_snow_factor is not None:
            manual_snow_factor = f"{schedule.manual_snow_factor:g}"
        if manual_snow_dates == "" and schedule.manual_snow_dates:
            manual_snow_dates = schedule.manual_snow_dates
        if horizon_mode == "":
            horizon_mode = schedule.horizon_mode or "weekday_calendar"
    if horizon_mode == "":
        horizon_mode = "weekday_calendar"
    schedule_form = ForecastScheduleForm(
        initial={
            "enabled": schedule.enabled if schedule else False,
            "start_at": (
                timezone.localtime(schedule.start_at).strftime("%Y-%m-%dT%H:%M")
                if schedule and schedule.start_at
                else ""
            ),
            "run_time": schedule.run_time.strftime("%H:%M") if schedule else "06:00",
            "days": schedule.days if schedule else days,
            "horizon_mode": schedule.horizon_mode if schedule else horizon_mode,
            "providers": (schedule.providers.split(",") if schedule and schedule.providers else selected_providers),
            "emails": schedule.emails if schedule else request.GET.get("emails", ""),
            "manual_snow_enable": schedule.manual_snow_enable if schedule else manual_snow_enable,
            "manual_snow_factor": schedule.manual_snow_factor if schedule else manual_snow_factor,
            "manual_snow_dates": schedule.manual_snow_dates if schedule else manual_snow_dates,
        }
    )
    from_s = request.GET.get("from") or ""
    to_s = request.GET.get("to") or ""
    date_s = request.GET.get("date") or ""
    dt_from = _parse_date(from_s)
    dt_to = _parse_date(to_s)
    dt_date = _parse_date(date_s)

    qs = SolarForecast.objects.filter(station=st, forecast_scope=forecast_scope).order_by("timestamp")
    if dt_from:
        qs = qs.filter(timestamp__gte=dt_from)
    if dt_to:
        qs = qs.filter(timestamp__lte=dt_to)
    if dt_date:
        qs = qs.filter(timestamp__date=dt_date.date())

    forecasts_raw = list(qs)

    daily = defaultdict(lambda: {"pred_final": 0.0, "pred_np": 0.0, "pred_xgb": 0.0, "pred_heur": 0.0})
    for f in forecasts_raw:
        ts = _localize_timestamp(f.timestamp)
        if ts is None:
            continue
        day_key = ts.date()
        if f.pred_final is not None:
            daily[day_key]["pred_final"] += float(f.pred_final)
        if f.pred_np is not None:
            daily[day_key]["pred_np"] += float(f.pred_np)
        if f.pred_xgb is not None:
            daily[day_key]["pred_xgb"] += float(f.pred_xgb)
        if f.pred_heur is not None:
            daily[day_key]["pred_heur"] += float(f.pred_heur)

    forecasts = [
        {
            "date": day,
            "pred_final_mw": values["pred_final"] / 1000.0,
            "pred_np_mw": values["pred_np"] / 1000.0,
            "pred_xgb_mw": values["pred_xgb"] / 1000.0,
            "pred_heur_mw": values["pred_heur"] / 1000.0,
        }
        for day, values in sorted(daily.items())
    ]

    return render(
        request,
        "dashboard/station_forecast_list.html",
        {
            "station": st,
            "forecasts": forecasts,
            "days": days,
            "selected_providers": selected_providers,
            "email_form": email_form,
            "manual_snow_enable": manual_snow_enable,
            "manual_snow_factor": manual_snow_factor,
            "manual_snow_dates": manual_snow_dates,
            "target_dates_raw": target_dates_raw,
            "manual_auto_send": manual_auto_send,
            "open_meteo_only": open_meteo_only,
            "horizon_mode": horizon_mode,
            "forecast_scope": forecast_scope,
            "schedule_form": schedule_form,
            "from": from_s,
            "to": to_s,
            "date": date_s,
            "count": len(forecasts_raw),
        },
    )


@login_required
def station_forecast_run(request, pk: int):
    st = _get_station_or_404(request.user, pk)
    if not _ensure_station_write_access(request, st):
        return redirect("dashboard:station-detail", pk=st.pk)
    days = int(request.GET.get("days", "7") or 7)
    providers = request.GET.getlist("providers") or None
    emails_raw = request.GET.get("emails", "")
    open_meteo_only = request.GET.get("open_meteo_only") in {"1", "true", "on", "yes"}
    horizon_mode = request.GET.get("horizon_mode") or ""
    manual_snow_enable = request.GET.get("manual_snow_enable") in {"1", "true", "on", "yes"}
    manual_snow_factor_raw = request.GET.get("manual_snow_factor")
    manual_snow_dates_raw = request.GET.get("manual_snow_dates") or ""
    target_dates_raw = request.GET.get("target_dates") or ""
    manual_auto_send = request.GET.get("manual_auto_send") in {"1", "true", "on", "yes"}
    forecast_scope = _normalize_forecast_scope(request.GET.get("scope") or "test")
    schedule = ForecastSchedule.objects.filter(station=st).first()
    if schedule:
        if not manual_snow_enable and request.GET.get("manual_snow_enable") is None:
            manual_snow_enable = schedule.manual_snow_enable
        if manual_snow_factor_raw in (None, "") and schedule.manual_snow_factor is not None:
            manual_snow_factor_raw = f"{schedule.manual_snow_factor:g}"
        if manual_snow_dates_raw == "" and schedule.manual_snow_dates:
            manual_snow_dates_raw = schedule.manual_snow_dates
        if horizon_mode == "":
            horizon_mode = schedule.horizon_mode or "weekday_calendar"
    if horizon_mode == "":
        horizon_mode = "weekday_calendar"
    if (not open_meteo_only) and schedule and schedule.providers:
        schedule_providers = [p.strip() for p in schedule.providers.split(",") if p.strip()]
        if "open_meteo_only" in schedule_providers:
            open_meteo_only = True
            providers = ["open_meteo"]
    manual_snow_factor = None
    if manual_snow_factor_raw not in (None, ""):
        try:
            manual_snow_factor = float(manual_snow_factor_raw)
        except ValueError:
            manual_snow_factor = None
    manual_snow_dates = []
    if manual_snow_dates_raw:
        for value in manual_snow_dates_raw.split(","):
            value = value.strip()
            if not value:
                continue
            parsed = _parse_date(value)
            if parsed:
                manual_snow_dates.append(parsed.date())

    target_dates = []
    if target_dates_raw:
        for value in target_dates_raw.split(","):
            value = value.strip()
            if not value:
                continue
            parsed = _parse_date(value)
            if parsed:
                target_dates.append(parsed.date())
    target_dates = sorted(set(target_dates))
    run_days = 1 if target_dates else days

    try:
        if open_meteo_only:
            providers = ["open_meteo"]
        res = run_forecast_for_station(
            st.pk,
            days=run_days,
            providers=providers,
            manual_snow_enable=manual_snow_enable,
            manual_snow_factor=manual_snow_factor,
            manual_snow_dates=manual_snow_dates,
            use_models=not open_meteo_only,
            horizon_mode=horizon_mode,
            forecast_scope=forecast_scope,
            target_dates=target_dates,
        )
        if res.get("ok"):
            actual_days = res.get("days") or run_days
            msg = f"Прогноз построен: {res.get('count')} строк, days={actual_days}, weather={res.get('weather_source')}, scope={forecast_scope}"
            if target_dates:
                msg += " | режим: фиксированные даты (параметр days игнорируется)"
            if open_meteo_only:
                msg += " | режим: Open-Meteo без истории"
            report = build_forecast_report(
                station=st,
                days=run_days,
                weather_source=res.get("weather_source"),
                recipients=[emails_raw],
                forecast_scope=forecast_scope,
                target_dates=res.get("target_dates") or [],
            )
            msg += f" | Отчёт сохранён: {report.file.name}"
            if manual_auto_send and emails_raw:
                if send_report_email(report, [emails_raw], st.name, days):
                    msg += f" | Email: {emails_raw}"
                else:
                    msg += " | Email: ошибка отправки"
            elif emails_raw:
                msg += " | Email: авто-отправка выключена"
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

    query = urlencode(
        {
            "days": days,
            "providers": providers or [],
            "emails": emails_raw,
            "manual_snow_enable": "1" if manual_snow_enable else "",
            "manual_snow_factor": manual_snow_factor_raw or "",
            "manual_snow_dates": manual_snow_dates_raw,
            "target_dates": target_dates_raw,
            "manual_auto_send": "1" if manual_auto_send else "",
            "open_meteo_only": "1" if open_meteo_only else "",
            "horizon_mode": horizon_mode,
            "scope": forecast_scope,
        },
        doseq=True,
    )
    return redirect(f"{reverse('dashboard:station-forecast-list', kwargs={'pk': st.pk})}?{query}")


@login_required
def station_forecast_schedule_update(request, pk: int):
    st = _get_station_or_404(request.user, pk)
    if not _ensure_station_write_access(request, st):
        return redirect("dashboard:station-detail", pk=st.pk)
    if request.method != "POST":
        return redirect("dashboard:station-forecast-list", pk=st.pk)

    form = ForecastScheduleForm(request.POST)
    if not form.is_valid():
        messages.error(request, "Ошибка в настройках автопрогноза.")
        return redirect("dashboard:station-forecast-list", pk=st.pk)

    schedule, _ = ForecastSchedule.objects.get_or_create(station=st)
    schedule.enabled = form.cleaned_data["enabled"]
    start_at = form.cleaned_data.get("start_at")
    if start_at and timezone.is_naive(start_at):
        start_at = timezone.make_aware(start_at, timezone.get_current_timezone())
    schedule.start_at = start_at
    schedule.run_time = form.cleaned_data["run_time"]
    schedule.days = form.cleaned_data["days"]
    schedule.horizon_mode = form.cleaned_data.get("horizon_mode") or "weekday_calendar"
    schedule.providers = ",".join(form.cleaned_data.get("providers") or [])
    schedule.emails = form.cleaned_data.get("emails", "")
    schedule.manual_snow_enable = form.cleaned_data.get("manual_snow_enable", False)
    schedule.manual_snow_factor = form.cleaned_data.get("manual_snow_factor") or 1.0
    schedule.manual_snow_dates = form.cleaned_data.get("manual_snow_dates", "")
    schedule.save()

    messages.success(request, "Настройки автопрогноза сохранены.")
    return redirect("dashboard:station-forecast-list", pk=st.pk)


@login_required
def station_forecast_scheduler_tick(request):
    force = request.GET.get("force") in {"1", "true", "on", "yes"}
    auto_history_rows = int(run_auto_history_updates() or 0)
    forecast_count = run_scheduled_forecasts(force=force)
    return JsonResponse(
        {
            "ok": True,
            "force": force,
            "auto_history_rows": auto_history_rows,
            "forecast_count": forecast_count,
            "count": forecast_count,
        }
    )


@login_required
def station_forecast_clear(request, pk: int):
    st = _get_station_or_404(request.user, pk)
    if not _ensure_station_write_access(request, st):
        return redirect("dashboard:station-detail", pk=st.pk)
    scope = _normalize_forecast_scope(request.POST.get("scope") or request.GET.get("scope") or "main")
    qs = SolarForecast.objects.filter(station=st, forecast_scope=scope)
    action = request.POST.get("action") or "all"

    if action == "before_date":
        before_date = _parse_date(request.POST.get("before_date") or "")
        if not before_date:
            messages.error(request, "Не удалось распознать дату удаления.")
            return redirect(f"{reverse('dashboard:station-forecast-list', kwargs={'pk': st.pk})}?scope={scope}")
        deleted, _ = qs.filter(timestamp__date__lt=before_date.date()).delete()
        messages.success(request, f"Удалено строк старого прогноза: {deleted} (до {before_date.date()}).")
        return redirect(f"{reverse('dashboard:station-forecast-list', kwargs={'pk': st.pk})}?scope={scope}")

    if action == "date_range":
        from_date = _parse_date(request.POST.get("from_date") or "")
        to_date = _parse_date(request.POST.get("to_date") or "")
        if not from_date or not to_date:
            messages.error(request, "Не удалось распознать диапазон дат удаления.")
            return redirect(f"{reverse('dashboard:station-forecast-list', kwargs={'pk': st.pk})}?scope={scope}")
        from_d = from_date.date()
        to_d = to_date.date()
        if from_d > to_d:
            from_d, to_d = to_d, from_d
        deleted, _ = qs.filter(timestamp__date__gte=from_d, timestamp__date__lte=to_d).delete()
        messages.success(request, f"Удалено строк прогноза: {deleted} (с {from_d} по {to_d}).")
        return redirect(f"{reverse('dashboard:station-forecast-list', kwargs={'pk': st.pk})}?scope={scope}")

    qs.delete()
    messages.success(request, "Прогноз очищен полностью.")
    return redirect(f"{reverse('dashboard:station-forecast-list', kwargs={'pk': st.pk})}?scope={scope}")


@login_required
def station_forecast_export(request, pk: int):
    st = _get_station_or_404(request.user, pk)

    from_s = request.GET.get("from") or ""
    to_s = request.GET.get("to") or ""
    date_s = request.GET.get("date") or ""
    dt_from = _parse_date(from_s)
    dt_to = _parse_date(to_s)
    dt_date = _parse_date(date_s)
    scope = _normalize_forecast_scope(request.GET.get("scope") or "main")

    qs = SolarForecast.objects.filter(station=st, forecast_scope=scope).order_by("timestamp")
    if dt_from:
        qs = qs.filter(timestamp__gte=dt_from)
    if dt_to:
        qs = qs.filter(timestamp__lte=dt_to)
    if dt_date:
        qs = qs.filter(timestamp__date=dt_date.date())

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
            "snowfall_fc",
            "snowdepth_fc",
            "weather_code_fc",
            "auto_snow_flag",
            "auto_fog_flag",
            "auto_winter_factor",
            "manual_snow_factor",
            "winter_factor_applied",
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
                "snowfall_fc",
                "snowdepth_fc",
                "weather_code_fc",
                "auto_snow_flag",
                "auto_fog_flag",
                "auto_winter_factor",
                "manual_snow_factor",
                "winter_factor_applied",
                "pred_final",
            ]
        )

    if "timestamp" in df.columns and not df.empty:
        ts = df["timestamp"].apply(_localize_timestamp)
        df["timestamp"] = _excel_safe_datetime(ts)
        for col in ["pred_np", "pred_xgb", "pred_heur", "pred_final"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce") / 1000.0
                df.rename(columns={col: f"{col}_mw"}, inplace=True)

    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="forecast")
    out.seek(0)

    resp = HttpResponse(
        out.getvalue(),
        content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    resp["Content-Disposition"] = f'attachment; filename="forecast_station_{st.pk}_mw.xlsx"'
    return resp
