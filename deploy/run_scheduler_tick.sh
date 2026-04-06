#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/forecast_platform}"
VENV_ACTIVATE="${VENV_ACTIVATE:-$PROJECT_DIR/venv/bin/activate}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$PROJECT_DIR"

if [[ -f "$VENV_ACTIVATE" ]]; then
  # shellcheck disable=SC1090
  source "$VENV_ACTIVATE"
fi

exec "$PYTHON_BIN" manage.py run_scheduled_forecasts
