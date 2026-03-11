#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/forecast_platform}"
VENV_PATH="${VENV_PATH:-venv/bin/activate}"
GUNICORN_PID_FILE_DEFAULT="${GUNICORN_PID_FILE_DEFAULT:-/run/gunicorn.pid}"
GUNICORN_PORT_DEFAULT="${GUNICORN_PORT_DEFAULT:-8000}"

cd "$PROJECT_DIR"

echo "[deploy] Pulling latest changes (rebase)..."
git pull --rebase

echo "[deploy] Activating virtualenv: $VENV_PATH"
# shellcheck disable=SC1090
source "$VENV_PATH"

echo "[deploy] Applying migrations..."
python3 manage.py migrate

if [ -f "$GUNICORN_PID_FILE_DEFAULT" ]; then
  echo "[deploy] Restarting via pid file: $GUNICORN_PID_FILE_DEFAULT"
  GUNICORN_PID_FILE="$GUNICORN_PID_FILE_DEFAULT" bash deploy/restart_portal.sh
else
  echo "[deploy] PID file not found, restarting via port: $GUNICORN_PORT_DEFAULT"
  GUNICORN_PORT="$GUNICORN_PORT_DEFAULT" bash deploy/restart_portal.sh
fi

echo "[deploy] Done."
