from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from django.apps import apps
from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import get_object_or_404, redirect, render


def _get_station_model():
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
    raise RuntimeError("Не нашёл модель станции (stations.Station или аналог).")


def _start_station_training_subprocess(station_id: int) -> tuple[bool, str]:
    base_dir = Path(getattr(settings, "BASE_DIR", "."))
    model_dir = Path(getattr(settings, "MODEL_DIR", base_dir / "models_cache"))
    model_dir.mkdir(parents=True, exist_ok=True)
    log_path = model_dir / f"train_station_{station_id}.log"

    cmd = [sys.executable, "manage.py", "train_station_models", str(station_id)]
    try:
        with log_path.open("ab") as log_file:
            subprocess.Popen(
                cmd,
                cwd=str(base_dir),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
    except Exception as exc:
        return False, str(exc)
    return True, str(log_path)


@login_required
def station_train_models(request, pk: int):
    Station = _get_station_model()
    st = get_object_or_404(Station, pk=pk)

    if request.method == "POST":
        ok, details = _start_station_training_subprocess(st.pk)
        if ok:
            messages.success(request, f"Обучение запущено в фоне. Лог: {details}")
        else:
            messages.error(request, f"Не удалось запустить обучение: {details}")

        return redirect("dashboard:station-detail", pk=st.pk)

    return render(request, "dashboard/station_train.html", {"station": st})
