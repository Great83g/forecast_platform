#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/forecast_platform}"
VENV_PATH="${VENV_PATH:-venv/bin/activate}"
GUNICORN_PID_FILE_DEFAULT="${GUNICORN_PID_FILE_DEFAULT:-/run/gunicorn.pid}"
GUNICORN_PORT_DEFAULT="${GUNICORN_PORT_DEFAULT:-8000}"
AUTO_STASH="${AUTO_STASH:-0}"

cd "$PROJECT_DIR"

echo "[deploy] Current commit: $(git rev-parse --short HEAD)"

if ! git diff --quiet || ! git diff --cached --quiet; then
  if [ "$AUTO_STASH" = "1" ]; then
    echo "[deploy] Local changes detected -> stashing (including untracked files)"
    git stash push -u -m "deploy-auto-stash $(date +%Y%m%d-%H%M%S)"
  else
    echo "[deploy] ERROR: local changes detected."
    echo "[deploy] Commit/stash changes first, or run with AUTO_STASH=1"
    echo "[deploy] Hint: git stash push -u && git pull --rebase --autostash"
    exit 1
  fi
fi

echo "[deploy] Pulling latest changes (rebase + autostash)..."
git pull --rebase --autostash

echo "[deploy] Activating virtualenv: $VENV_PATH"
# shellcheck disable=SC1090
source "$VENV_PATH"

echo "[deploy] Planned migrations..."
python3 manage.py migrate --plan

echo "[deploy] Applying migrations..."
python3 manage.py migrate

if [ -f "$GUNICORN_PID_FILE_DEFAULT" ]; then
  echo "[deploy] Restarting via pid file: $GUNICORN_PID_FILE_DEFAULT"
  GUNICORN_PID_FILE="$GUNICORN_PID_FILE_DEFAULT" bash deploy/restart_portal.sh
else
  echo "[deploy] PID file not found, restarting via port: $GUNICORN_PORT_DEFAULT"
  GUNICORN_PORT="$GUNICORN_PORT_DEFAULT" bash deploy/restart_portal.sh
fi

echo "[deploy] Smoke check:"
curl -I "http://127.0.0.1:${GUNICORN_PORT_DEFAULT}/" || true

echo "[deploy] Done. Current commit: $(git rev-parse --short HEAD)"
