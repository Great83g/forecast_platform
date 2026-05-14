#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/forecast_platform}"
VENV_PATH="${VENV_PATH:-$PROJECT_DIR/venv/bin/activate}"
GUNICORN_PID_FILE_DEFAULT="${GUNICORN_PID_FILE_DEFAULT:-/run/gunicorn.pid}"
GUNICORN_PORT_DEFAULT="${GUNICORN_PORT_DEFAULT:-8000}"
RUN_TESTS="${RUN_TESTS:-1}"
EXPECTED_MARKER="postfactum-forecast-panel"

cd "$PROJECT_DIR"

if [ -f "$VENV_PATH" ]; then
  # shellcheck disable=SC1090
  source "$VENV_PATH"
else
  echo "[wind-forecast-runs] ERROR: virtualenv activate script not found: $VENV_PATH" >&2
  exit 1
fi

echo "[wind-forecast-runs] Fetching and forcing main to origin/main"
git fetch --all --tags --prune
git checkout main
git reset --hard origin/main

echo "[wind-forecast-runs] Current commit: $(git rev-parse --short HEAD) $(git log -1 --pretty=%s)"

if ! rg -q "$EXPECTED_MARKER" wind/templates/wind/station_forecast_list.html; then
  echo "[wind-forecast-runs] ERROR: expected marker '$EXPECTED_MARKER' is not in wind/templates/wind/station_forecast_list.html" >&2
  echo "[wind-forecast-runs] You are probably on an old origin/main. Merge/pull the PR commit first, then rerun this script." >&2
  exit 1
fi

echo "[wind-forecast-runs] Syntax checks"
python3 -m py_compile \
  manage.py \
  backend/urls.py \
  wind/models.py \
  wind/views.py \
  wind/services/forecast_runs.py \
  wind/services/forecasting.py \
  dashboard/services/forecast_scheduler.py \
  wind/tests.py
python3 -m compileall -q wind dashboard/services

echo "[wind-forecast-runs] Migrations"
python3 manage.py migrate

echo "[wind-forecast-runs] Static files"
python3 manage.py collectstatic --noinput

if [ "$RUN_TESTS" = "1" ]; then
  echo "[wind-forecast-runs] Tests"
  python3 manage.py test wind.tests.WindForecastModuleTests -v 2
else
  echo "[wind-forecast-runs] RUN_TESTS=0, skipping tests"
fi

echo "[wind-forecast-runs] Restarting web process"
if [ -f "$GUNICORN_PID_FILE_DEFAULT" ]; then
  sudo env GUNICORN_PID_FILE="$GUNICORN_PID_FILE_DEFAULT" bash deploy/restart_portal.sh
else
  sudo env GUNICORN_PORT="$GUNICORN_PORT_DEFAULT" PREFER_DIRECT_GUNICORN_RELOAD=1 bash deploy/restart_portal.sh
fi

echo "[wind-forecast-runs] Verifying deployed source marker"
rg -n "$EXPECTED_MARKER|data-postfactum-panel-version" wind/templates/wind/station_forecast_list.html

echo "[wind-forecast-runs] OK. Hard refresh browser (Ctrl+F5)."
