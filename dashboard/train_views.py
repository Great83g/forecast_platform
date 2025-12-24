from __future__ import annotations

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404, redirect, render

from django.apps import apps

from dashboard.services.train_models import train_models_for_station


def _get_station_model():
    # В твоей БД модель станции сейчас: stations.StationsStation (см. inspectdb)
    # Но на всякий случай ищем по кандидатам.
    for (app_label, model_name) in [
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
            train_models_for_station(st.pk)
            messages.success(request, "Обучение моделей запущено/выполнено.")
        except Exception as e:
            messages.error(request, f"Ошибка обучения: {e}")

        return redirect("dashboard:station-list")

    return render(request, "dashboard/station_train.html", {"station": st})
