#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENV_PATH="${VENV_PATH:-$PROJECT_DIR/venv/bin/activate}"
DEFAULT_GUNICORN_PID_FILE="${DEFAULT_GUNICORN_PID_FILE:-/run/gunicorn.pid}"
DEFAULT_GUNICORN_PORT="${DEFAULT_GUNICORN_PORT:-8000}"

cd "$PROJECT_DIR"

echo "[deploy] Checking unresolved merges..."
if git diff --name-only --diff-filter=U | grep -q .; then
  echo "[deploy] ERROR: there are unresolved merge conflicts."
  git diff --name-only --diff-filter=U
  echo "[deploy] Resolve conflicts, then run this script again."
  exit 1
fi

echo "[deploy] Pulling latest changes (rebase)..."
git pull --rebase

echo "[deploy] Activating virtualenv: $VENV_PATH"
# shellcheck disable=SC1090
source "$VENV_PATH"

echo "[deploy] Applying migrations..."
python3 manage.py migrate

if [ -n "${GUNICORN_PID_FILE:-}" ]; then
  echo "[deploy] Restarting via explicit GUNICORN_PID_FILE=$GUNICORN_PID_FILE"
  GUNICORN_PID_FILE="$GUNICORN_PID_FILE" bash deploy/restart_portal.sh
elif [ -f "$DEFAULT_GUNICORN_PID_FILE" ]; then
  echo "[deploy] Restarting via detected pid file: $DEFAULT_GUNICORN_PID_FILE"
  GUNICORN_PID_FILE="$DEFAULT_GUNICORN_PID_FILE" bash deploy/restart_portal.sh
else
  echo "[deploy] PID file not found, restarting via GUNICORN_PORT=$DEFAULT_GUNICORN_PORT"
  GUNICORN_PORT="$DEFAULT_GUNICORN_PORT" bash deploy/restart_portal.sh
fi

echo "[deploy] Done."
