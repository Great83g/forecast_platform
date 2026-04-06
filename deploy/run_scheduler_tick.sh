#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/forecast_platform}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$PROJECT_DIR/venv/bin/activate}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LOCK_FILE="${LOCK_FILE:-/tmp/forecast_scheduler_tick.lock}"

cd "$PROJECT_DIR"

if [[ -f "$VENV_ACTIVATE" ]]; then
  # shellcheck disable=SC1090
  source "$VENV_ACTIVATE"
fi

if command -v flock >/dev/null 2>&1; then
  exec flock -n "$LOCK_FILE" "$PYTHON_BIN" manage.py run_scheduled_forecasts
fi

exec "$PYTHON_BIN" manage.py run_scheduled_forecasts
