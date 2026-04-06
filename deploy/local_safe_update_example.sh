#!/usr/bin/env bash
cd ~/forecast_platform
source venv/bin/activate
set -euo pipefail

echo "=== SAFE UPDATE START $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="

# --- helper: есть ли commit в репо ---
has_commit() {
  git cat-file -e "$1^{commit}" 2>/dev/null
}

# --- helper: применить commit только если его ещё нет в HEAD ---
apply_if_missing() {
  local sha="$1"
  local label="$2"

  if ! has_commit "$sha"; then
    echo "[SKIP] $label ($sha) не найден локально после fetch."
    return 0
  fi

  if git merge-base --is-ancestor "$sha" HEAD; then
    echo "[OK] $label уже в текущей ветке."
    return 0
  fi

  echo "[APPLY] $label ($sha)"
  if ! git cherry-pick "$sha"; then
    echo "[ERR] Конфликт при cherry-pick $sha. Разрули и запусти:"
    echo "      git cherry-pick --continue   # или git cherry-pick --abort"
    exit 1
  fi
}

# 0) stash локальных правок
git status --short
git stash push -u -m "wip-before-safe-update-$(date +%F_%H-%M-%S)" || true

# 1) обновление базы
git fetch --all --prune
git checkout main
git pull --ff-only origin main

# 2) нужные коммиты (wind scheduler/server update)
apply_if_missing f536e80 "Wind: scheduled forecasts + horizon_mode/calendar UI"
apply_if_missing 1053716 "Wind: refactor scheduler service + server runner script/docs"

# 3) вернуть stash (если есть)
if git stash list | grep -q "wip-before-safe-update"; then
  git stash pop || true
fi

# 4) деплой
python3 manage.py migrate
python3 manage.py cleanup_model_cache
python3 manage.py collectstatic --noinput

if [ -f /run/gunicorn.pid ]; then
  sudo env GUNICORN_PID_FILE=/run/gunicorn.pid bash deploy/restart_portal.sh
else
  sudo env GUNICORN_PORT=8000 bash deploy/restart_portal.sh
fi

# 5) (опционально) один тик шедулера сразу после релиза
bash deploy/run_scheduler_tick.sh

echo "=== SAFE UPDATE DONE ==="
git log --oneline -n 10

