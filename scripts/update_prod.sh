#!/usr/bin/env bash
set -euo pipefail

cd ~/forecast_platform

echo "=== 0) Активировать venv ==="
source venv/bin/activate

echo "=== 1) Обновить main ==="
git fetch --all --tags --prune
git checkout main
git pull --ff-only origin main

echo "=== 2) Проверка коммитов ==="
git log --oneline -n 5

echo "=== 2.1) Установка зависимостей (если есть requirements.txt) ==="
if [ -f requirements.txt ]; then
  pip install -r requirements.txt
else
  echo "requirements.txt не найден — пропускаю pip install"
fi

echo "=== 3) Миграции ==="
python manage.py migrate --noinput

echo "=== 4) Статика ==="
python manage.py collectstatic --noinput

echo "=== 4.1) Django check ==="
python manage.py check

echo "=== 5) Рестарт backend-сервиса ==="
sudo systemctl restart forecast_portal.service
sudo systemctl status forecast_portal.service --no-pager -l | tail -n 60
