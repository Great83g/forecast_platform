# dashboard/views.py

from io import BytesIO
from datetime import datetime

import pandas as pd
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.utils.dateformat import format
from django.utils.text import slugify
from openpyxl import Workbook

from stations.models import Station
from solar.models import SolarRecord, SolarForecast
from .forms import StationForm, UploadHistoryForm
from .services.train_models import train_models_for_station
from .services.forecast_engine import run_forecast_for_station


# ========= ОБУЧЕНИЕ МОДЕЛЕЙ =========

@login_required
def station_train_models(request, pk):
    """
    Кнопка на станции: обучить/переобучить модели по истории.
    """
    station = get_object_or_404(Station, pk=pk)

    try:
        n_rows, np_path, xgb_path = train_models_for_station(station)
        if n_rows == 0:
            messages.warning(
                request,
                "Нет исторических данных по станции — обучать нечего.",
            )
        else:
            np_name = np_path.name if np_path is not None else "не создан"
            xgb_name = xgb_path.name if xgb_path is not None else "не создан"

            messages.success(
                request,
                f"Модели обучены по {n_rows} строкам истории. "
                f"NP: {np_name}, XGB: {xgb_name}."
            )
    except Exception as e:
        messages.error(request, f"Ошибка при обучении моделей: {e}")

    return redirect("dashboard-station-detail", pk=station.pk)


# ========= СПИСОК / СОЗДАНИЕ / ДЕТАЛКА =========

@login_required
def station_list(request):
    """Список всех станций."""
    stations = Station.objects.select_related("org").all()
    return render(request, "dashboard/station_list.html", {
        "stations": stations,
    })


@login_required
def station_create(request):
    """Создание новой станции через форму."""
    if request.method == "POST":
        form = StationForm(request.POST)
        if form.is_valid():
            station = form.save()
            messages.success(request, "Станция создана.")
            return redirect("dashboard-station-detail", pk=station.pk)
    else:
        form = StationForm()

    return render(request, "dashboard/station_create.html", {
        "form": form,
    })


@login_required
def station_detail(request, pk):
    """Деталка станции + последние 7 дней для графика."""
    station = get_object_or_404(Station, pk=pk)

    # последние 7 дней (168 часов)
    records_qs = (
        SolarRecord.objects
        .filter(station=station)
        .order_by("-timestamp")[:168]
    )
    # приводим в хронологический порядок
    records = list(records_qs)[::-1]

    timestamps = [format(r.timestamp, "Y-m-d H:i:s") for r in records]
    power = [r.power_kw for r in records]
    irr = [r.irradiation for r in records]
    air_temp = [r.air_temp for r in records]
    pv_temp = [r.pv_temp for r in records]

    return render(request, "dashboard/station_detail.html", {
        "station": station,
        "records": records,
        "timestamps": timestamps,
        "power": power,
        "irr": irr,
        "air_temp": air_temp,
        "pv_temp": pv_temp,
    })


# ========= ИСТОРИЯ: ЗАГРУЗКА / ОЧИСТКА / УДАЛЕНИЕ =========

@login_required
def station_upload_history(request, pk):
    """
    Страница истории станции:
    - загрузка CSV/Excel
    - очистка всей истории
    - удаление выбранных записей
    - фильтр по датам (GET ?from=YYYY-MM-DD&to=YYYY-MM-DD)
    """
    station = get_object_or_404(Station, pk=pk)

    # ---------- ФИЛЬТР ДАТ ----------
    from_date = request.GET.get("from") or ""
    to_date = request.GET.get("to") or ""

    base_qs = SolarRecord.objects.filter(station=station)
    total_count = base_qs.count()

    history_qs = base_qs.order_by("timestamp")
    if from_date:
        history_qs = history_qs.filter(timestamp__date__gte=from_date)
    if to_date:
        history_qs = history_qs.filter(timestamp__date__lte=to_date)

    history = history_qs
    history_count = history.count()

    # ---------- POST-действия ----------
    if request.method == "POST":
        action = request.POST.get("action", "")

        # === ЗАГРУЗКА ФАЙЛА ===
        if action == "upload":
            form = UploadHistoryForm(request.POST, request.FILES)
            if form.is_valid():
                file = form.cleaned_data["file"]
                filename = file.name.lower()

                # 1) читаем CSV / Excel
                try:
                    if filename.endswith(".csv"):
                        df = pd.read_csv(file)
                    elif filename.endswith((".xlsx", ".xls")):
                        df = pd.read_excel(file)
                    else:
                        messages.error(request, "Поддерживаются только файлы .csv или .xlsx")
                        return redirect("dashboard-station-upload", pk=station.pk)
                except Exception as e:
                    messages.error(request, f"Ошибка чтения файла: {e}")
                    return redirect("dashboard-station-upload", pk=station.pk)

                # 2) нормализуем заголовки (убираем пробелы) и приводим Power_KW -> Power_kW
                df.columns = [str(c).strip() for c in df.columns]
                if "Power_KW" in df.columns and "Power_kW" not in df.columns:
                    df.rename(columns={"Power_KW": "Power_kW"}, inplace=True)

                # 3) проверяем колонки
                required_cols = ["ds", "Irradiation", "Air_Temp", "PV_Temp", "Power_kW"]
                missing = [c for c in required_cols if c not in df.columns]
                if missing:
                    messages.error(request, f"Отсутствуют колонки: {missing}")
                    return redirect("dashboard-station-upload", pk=station.pk)

                # 4) парсим даты
                try:
                    df["ds"] = pd.to_datetime(df["ds"], dayfirst=True, errors="coerce")
                    if df["ds"].isnull().any():
                        messages.error(
                            request,
                            "Некоторые значения 'ds' не распознаны как даты. "
                            "Убедитесь, что формат даты — YYYY-MM-DD HH:MM или DD.MM.YYYY HH:MM.",
                        )
                        return redirect("dashboard-station-upload", pk=station.pk)
                except Exception as e:
                    messages.error(request, f"Не удалось распарсить колонку 'ds' как дату: {e}")
                    return redirect("dashboard-station-upload", pk=station.pk)

                # 5) пишем в базу (update_or_create -> дублей не будет)
                total_rows = 0
                created_rows = 0

                for _, row in df.iterrows():
                    total_rows += 1
                    _, created_flag = SolarRecord.objects.update_or_create(
                        station=station,
                        timestamp=row["ds"],
                        defaults={
                            "irradiation": row["Irradiation"],
                            "air_temp": row["Air_Temp"],
                            "pv_temp": row["PV_Temp"],
                            "power_kw": row["Power_kW"],
                        },
                    )
                    if created_flag:
                        created_rows += 1

                messages.success(
                    request,
                    f"Обработано строк: {total_rows}. Новых записей: {created_rows}."
                )
                return redirect("dashboard-station-upload", pk=station.pk)
            else:
                messages.error(request, "Форма загрузки заполнена некорректно.")
                return redirect("dashboard-station-upload", pk=station.pk)

        # === ОЧИСТКА ВСЕЙ ИСТОРИИ ===
        elif action == "clear":
            deleted, _ = base_qs.delete()
            messages.success(request, f"История очищена, удалено записей: {deleted}.")
            return redirect("dashboard-station-upload", pk=station.pk)

        # === УДАЛЕНИЕ ВЫБРАННЫХ ===
        elif action == "delete_selected":
            selected_ids = request.POST.getlist("selected")
            if not selected_ids:
                messages.warning(request, "Не выбраны записи для удаления.")
                return redirect("dashboard-station-upload", pk=station.pk)

            deleted, _ = base_qs.filter(id__in=selected_ids).delete()
            messages.success(request, f"Удалено записей: {deleted}.")
            return redirect("dashboard-station-upload", pk=station.pk)

    else:
        form = UploadHistoryForm()

    return render(request, "dashboard/station_upload.html", {
        "station": station,
        "form": form,
        "history": history,
        "history_count": history_count,
        "total_count": total_count,
        "from_date": from_date,
        "to_date": to_date,
    })


# ========= ЭКСПОРТ ИСТОРИИ =========

@login_required
def station_export_history(request, pk):
    """
    Экспорт истории станции в Excel.
    Учитывает фильтр ?from=YYYY-MM-DD&to=YYYY-MM-DD.
    """
    station = get_object_or_404(Station, pk=pk)

    from_date = request.GET.get("from") or ""
    to_date = request.GET.get("to") or ""

    qs = SolarRecord.objects.filter(station=station).order_by("timestamp")
    if from_date:
        qs = qs.filter(timestamp__date__gte=from_date)
    if to_date:
        qs = qs.filter(timestamp__date__lte=to_date)

    rows = list(
        qs.values(
            "timestamp",
            "irradiation",
            "air_temp",
            "pv_temp",
            "power_kw",
        )
    )

    if not rows:
        messages.warning(request, "Нет данных для экспорта.")
        return redirect("dashboard-station-upload", pk=station.pk)

    # --- формируем DataFrame ---
    df = pd.DataFrame(rows)

    df.rename(
        columns={
            "timestamp": "ds",
            "irradiation": "Irradiation",
            "air_temp": "Air_Temp",
            "pv_temp": "PV_Temp",
            "power_kw": "Power_kW",
        },
        inplace=True,
    )

    df["ds"] = pd.to_datetime(df["ds"])
    try:
        df["ds"] = df["ds"].dt.tz_localize(None)
    except TypeError:
        pass

    df = df.fillna(0)

    for col in ["Irradiation", "Air_Temp", "PV_Temp", "Power_KW"]:
        df[col] = df[col].astype(float).round(6)

    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="History", index=False)
    output.seek(0)

    base_name = slugify(station.name or f"station-{station.pk}")
    if from_date or to_date:
        suffix = f"_{from_date or ''}_{to_date or ''}".replace("--", "-")
    else:
        suffix = ""
    filename = f"{base_name}_history{suffix}.xlsx"

    response = HttpResponse(
        output.getvalue(),
        content_type=(
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        ),
    )
    response["Content-Disposition"] = f'attachment; filename="{filename}"'
    return response


# ========= НОВАЯ ВЬЮХА ДЛЯ КНОПКИ «СДЕЛАТЬ ПРОГНОЗ» =========

@login_required
def run_station_forecast(request, pk):
    """
    Запускает прогноз для станции (история -> модели) и сохраняет в SolarForecast.
    """
    station = get_object_or_404(Station, pk=pk)

    try:
        rows = run_forecast_for_station(station, days=3)
        messages.success(
            request,
            f"Прогноз обновлён, в базе сохранено записей: {rows}."
        )
    except Exception as e:
        messages.error(request, f"Ошибка при запуске прогноза: {e}")

    return redirect("dashboard-station-detail", pk=station.pk)


# ========= ПРОСМОТР + ЭКСПОРТ ПРОГНОЗА =========

@login_required
def station_forecast_list(request, pk):
    """
    Просмотр прогноза SolarForecast с фильтром по датам.
    """
    station = get_object_or_404(Station, pk=pk)

    from_date = request.GET.get("from") or ""
    to_date = request.GET.get("to") or ""

    base_qs = SolarForecast.objects.filter(station=station).order_by("timestamp")
    total_count = base_qs.count()

    qs = base_qs
    if from_date:
        qs = qs.filter(timestamp__date__gte=from_date)
    if to_date:
        qs = qs.filter(timestamp__date__lte=to_date)

    forecasts = qs
    filtered_count = forecasts.count()

    return render(request, "dashboard/station_forecast.html", {
        "station": station,
        "forecasts": forecasts,
        "total_count": total_count,
        "filtered_count": filtered_count,
        "from_date": from_date,
        "to_date": to_date,
    })


@login_required
def station_forecast_clear(request, pk):
    """
    Очищает все сохранённые прогнозы станции.
    """
    station = get_object_or_404(Station, pk=pk)

    if request.method != "POST":
        messages.error(request, "Неверный метод запроса для очистки прогноза.")
        return redirect("dashboard-station-forecast-list", pk=station.pk)

    deleted, _ = SolarForecast.objects.filter(station=station).delete()

    if deleted:
        messages.success(request, f"Удалено записей прогноза: {deleted}.")
    else:
        messages.warning(request, "Нет прогнозов для удаления.")
    return redirect("dashboard-station-forecast-list", pk=station.pk)


@login_required
def station_export_forecast(request, pk):
    """
    Выгрузка прогноза станции в Excel.
    ВРЕМЯ конвертируем в локальный часовой пояс (settings.TIME_ZONE)
    и отбрасываем часы вне диапазона 06:00–20:00.
    """

    station = get_object_or_404(Station, pk=pk)

    qs = SolarForecast.objects.filter(
        station=station,
    ).order_by("timestamp")

    # Фильтры по датам из формы (если есть)
    date_from = request.GET.get("date_from")  # формат mm/dd/yyyy
    date_to = request.GET.get("date_to")      # формат mm/dd/yyyy

    if date_from:
        dt_from = datetime.strptime(date_from, "%m/%d/%Y").date()
        qs = qs.filter(timestamp__date__gte=dt_from)

    if date_to:
        dt_to = datetime.strptime(date_to, "%m/%d/%Y").date()
        qs = qs.filter(timestamp__date__lte=dt_to)

    # === Готовим Excel ===
    wb = Workbook()
    ws = wb.active
    ws.title = "Forecast"

    # Шапка
    ws.append(["ds", "Pred_NP", "Pred_XGB", "Pred_Heur", "Pred_Final"])

    local_tz = timezone.get_default_timezone()

    for rec in qs.iterator():
        # Переводим время из UTC -> локальное, убираем tzinfo, чтобы Excel
        # видел обычный наивный datetime в местном времени
        ts_local = timezone.localtime(rec.timestamp, local_tz).replace(tzinfo=None)

        # Оставляем только дневные часы (06–20), как на сайте
        if ts_local.hour < 6 or ts_local.hour > 20:
            continue

        ws.append([
            ts_local,
            float(rec.pred_np or 0.0),
            float(rec.pred_xgb or 0.0),
            float(rec.pred_heur or 0.0),
            float(rec.pred_final or 0.0),
        ])

    # Сохраняем книгу в память
    buffer = BytesIO()
    wb.save(buffer)
    buffer.seek(0)

    # Отдаём файл пользователю
    filename = f"forecast_station_{station.pk}.xlsx"
    response = HttpResponse(
        buffer.getvalue(),
        content_type=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
    )
    response["Content-Disposition"] = f'attachment; filename="{filename}"'
    return response
