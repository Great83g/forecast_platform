# Обновление сервера для ветромодуля

Быстрый запуск одним скриптом:

```bash
cd ~/forecast_platform
bash deploy/apply_wind_module_update.sh
```

## Что делает скрипт

1. Активирует virtualenv (`venv/bin/activate`), если найден.
2. Сбрасывает изменения только в `models_cache`.
3. Обновляет `main` (`fetch` + `pull --ff-only`).
4. Выполняет Django-команды:
   - `python3 manage.py migrate`
   - `python3 manage.py cleanup_model_cache`
   - `python3 manage.py collectstatic --noinput`
5. Рестартует портал через `deploy/restart_portal.sh`.
6. Показывает `git status`.

## Эквивалент в ручном режиме

```bash
cd ~/forecast_platform
source venv/bin/activate
set -euo pipefail

git restore --worktree --staged models_cache || true
git fetch --all --prune
git checkout main
git pull --ff-only origin main

python3 manage.py migrate
python3 manage.py cleanup_model_cache
python3 manage.py collectstatic --noinput

if [ -f /run/gunicorn.pid ]; then
  sudo env GUNICORN_PID_FILE=/run/gunicorn.pid bash deploy/restart_portal.sh
else
  sudo env GUNICORN_PORT=8000 bash deploy/restart_portal.sh
fi

git status
```
