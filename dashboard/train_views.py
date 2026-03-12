from __future__ import annotations

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404, redirect, render

from django.apps import apps

from dashboard.services.forecast_engine import train_models_for_station


def _get_station_model():
    # Ищем рабочую модель станции среди известных вариантов.
    # В текущем проекте обычно используется stations.Station.
    for (app_label, model_name) in [
        ("stations", "Station"),
        ("stations", "StationsStation"),
        ("solar", "Station"),
        ("solar", "SolarStation"),
        ("solar", "SolarPlant"),
        ("solar", "SolarPowerStation"),
    ]:
        try:
            return apps.get_model(app_label, model_name)
        except Exception:
            continue
    raise RuntimeError("Не нашёл модель станции (ожидал stations.StationsStation или аналоги).")


@login_required
def station_train_models(request, pk: int):
    Station = _get_station_model()
    st = get_object_or_404(Station, pk=pk)

    if request.method == "POST":
        try:
            # обучение NP + XGB (реализовано в forecast_engine.py)
            train_models_for_station(st)
            messages.success(request, "Обучение моделей запущено/выполнено.")
        except Exception as e:
            messages.error(request, f"Ошибка обучения: {e}")

        return redirect("dashboard:station-detail", pk=st.pk)

    return render(request, "dashboard/station_train.html", {"station": st})
