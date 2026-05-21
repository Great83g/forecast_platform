# Деплой PR: AI Assistant

Скопируйте и выполните этот блок на сервере после merge PR в `main`.
Скрипт синхронизирует сервер с `origin/main`, проверяет миграции, применяет их,
собирает статику, перезапускает портал и делает минимальный smoke-check.

```bash
cd ~/forecast_platform
source venv/bin/activate
set -euo pipefail

echo "=== 1) Обновляем код с origin/main ==="
git fetch --all --tags --prune
git checkout main
git reset --hard origin/main
git log -1 --oneline

echo "=== 2) Проверяем, что рабочее дерево чистое ==="
git status --short

echo "=== 3) Проверяем, что миграции не забыты ==="
python3 manage.py makemigrations --check --dry-run

echo "=== 4) Применяем миграции БД ==="
python3 manage.py migrate

echo "=== 5) Собираем статику ==="
python3 manage.py collectstatic --noinput

echo "=== 6) Перезапускаем портал ==="
if [ -f /run/gunicorn.pid ]; then
  GUNICORN_PID_FILE=/run/gunicorn.pid bash deploy/restart_portal.sh
else
  GUNICORN_PORT=8000 bash deploy/restart_portal.sh
fi

echo "=== 7) Smoke-check главной страницы ==="
curl -I http://127.0.0.1:8000/ || true

echo "=== 8) Smoke-check AI Assistant API auth guard ==="
curl -I "http://127.0.0.1:8000/api/assistant/query/" || true

echo "=== DONE ==="
git log -1 --oneline
```

Ожидаемо, `/api/assistant/query/` доступен только авторизованным пользователям и только
методом `POST`, поэтому unauthenticated/head smoke-check нужен лишь для проверки, что URL
маршрутизируется порталом, а не для полноценного функционального запроса.


## Как не терять локально заменённые картинки guide после обновления

Если вы вручную меняете файлы в `dashboard/static/dashboard/img/guide/*.svg`,
то `git reset --hard origin/main` перезаписывает их из репозитория.

Используйте безопасный скрипт обновления:

```bash
cd ~/forecast_platform
source venv/bin/activate
bash deploy/update_portal_preserve_assets.sh
```

Что он делает:
1. Сохраняет ваши локальные `guide/*.svg` в `.local_assets_backup/guide`.
2. Обновляет код через `git fetch + git reset --hard`.
3. Возвращает сохранённые локальные картинки обратно.
4. Запускает `migrate`, `collectstatic`, `restart`, smoke-check.
