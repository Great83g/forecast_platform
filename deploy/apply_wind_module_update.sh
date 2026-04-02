#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/forecast_platform}"
VENV_PATH="${VENV_PATH:-$PROJECT_DIR/venv/bin/activate}"
GUNICORN_PID_FILE="${GUNICORN_PID_FILE:-/run/gunicorn.pid}"
GUNICORN_PORT="${GUNICORN_PORT:-8000}"

cd "$PROJECT_DIR"

if [ -f "$VENV_PATH" ]; then
  # shellcheck disable=SC1090
  source "$VENV_PATH"
else
  echo "[WARN] virtualenv activate script not found: $VENV_PATH"
fi

# 0) убрать локальные изменения только в models_cache
if [ -d models_cache ]; then
  git restore --worktree --staged models_cache || true
fi

# 1) обновить код

git fetch --all --prune
git checkout main
git pull --ff-only origin main

# 2) django шаги
python3 manage.py migrate
python3 manage.py cleanup_model_cache
python3 manage.py collectstatic --noinput

# 3) без обучения
# intentionally skipped

# 4) рестарт
if [ -f "$GUNICORN_PID_FILE" ]; then
  sudo env GUNICORN_PID_FILE="$GUNICORN_PID_FILE" bash deploy/restart_portal.sh
else
  sudo env GUNICORN_PORT="$GUNICORN_PORT" bash deploy/restart_portal.sh
fi

# 5) быстрый контроль

git status

echo "[OK] Wind module update steps completed."
