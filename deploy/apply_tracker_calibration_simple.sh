#!/usr/bin/env bash
set -euo pipefail

cd "${PROJECT_DIR:-$HOME/forecast_platform}"

echo "=== 0) Активировать venv ==="
source "${VENV_PATH:-venv/bin/activate}"

echo "=== 1) Обновить main ==="
git fetch --all --tags --prune
git checkout main
git pull --ff-only origin main

echo "=== 2) Проверка коммитов ==="
git log --oneline -n 5

echo "=== 3) Проверить, что код диагностики трекера на месте ==="
rg -n "compare_solar_forecast_modes|tracker_station_calibration_factor|FORECAST_COMPARE" \
  dashboard/management/commands/compare_solar_forecast_modes.py \
  dashboard/services/forecast_engine.py \
  dashboard/services/forecast_diagnostics.py

echo "=== 4) Быстрая проверка синтаксиса ==="
python3 -m py_compile \
  dashboard/services/forecast_engine.py \
  dashboard/services/forecast_diagnostics.py \
  dashboard/management/commands/compare_solar_forecast_modes.py

echo "=== 5) Проверить миграции ==="
python3 manage.py makemigrations --check --dry-run

echo "=== 6) Применить миграции ==="
python3 manage.py migrate

echo "=== 7) Собрать статику ==="
python3 manage.py collectstatic --noinput

echo "=== 8) Точечные тесты (можно пропустить: RUN_TESTS=0) ==="
if [ "${RUN_TESTS:-1}" = "1" ]; then
  python3 manage.py test \
    dashboard.tests.SingleAxisTrackerPostProcessingTests \
    dashboard.tests.TrackerMiddayFloorTests \
    dashboard.tests.ForecastModeDiagnosticsTests \
    -v 2
else
  echo "RUN_TESTS=0 — тесты пропущены"
fi

echo "=== 9) Перезапуск портала ==="
if [ -f "${GUNICORN_PID_FILE:-/run/gunicorn.pid}" ]; then
  sudo env GUNICORN_PID_FILE="${GUNICORN_PID_FILE:-/run/gunicorn.pid}" bash deploy/restart_portal.sh
else
  sudo env GUNICORN_PORT="${GUNICORN_PORT:-8000}" PREFER_DIRECT_GUNICORN_RELOAD=1 bash deploy/restart_portal.sh
fi

echo "=== 10) Диагностика Shu 100 MW за 25–31 мая (если есть данные) ==="
python3 manage.py compare_solar_forecast_modes \
  --station "${DIAG_STATION:-Shu 100 MW}" \
  --from "${DIAG_FROM:-2026-05-25}" \
  --to "${DIAG_TO:-2026-05-31}" \
  --threshold "${DIAG_THRESHOLD:-5}" || true

echo "=== OK ==="
git rev-parse --short HEAD
