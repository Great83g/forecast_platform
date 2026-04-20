#!/usr/bin/env bash
set -euo pipefail

cd ~/forecast_platform
source venv/bin/activate

# 0) убрать локальные изменения только в models_cache (частая история после cleanup)
git restore --worktree --staged models_cache || true

# 1) обновить код
git fetch --all --prune
git checkout main
git pull --ff-only origin main

# 2) django шаги
python3 manage.py migrate
python3 manage.py cleanup_model_cache
python3 manage.py collectstatic --noinput

# 3) без обучения

# 4) рестарт
if [ -f /run/gunicorn.pid ]; then
  sudo env GUNICORN_PID_FILE=/run/gunicorn.pid bash deploy/restart_portal.sh
else
  sudo env GUNICORN_PORT=8000 bash deploy/restart_portal.sh
fi

# 5) быстрый контроль
git status
