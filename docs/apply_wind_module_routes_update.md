# Как применить обновление (готовый скрипт)

Ниже даю **готовый bash-скрипт**, как вы просили, чтобы можно было копировать целиком в каждый пост.

```bash
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
```

## Вариант запуска одним файлом

```bash
cat > /tmp/apply_portal_update.sh <<'BASH'
#!/usr/bin/env bash
set -euo pipefail

cd ~/forecast_platform
source venv/bin/activate

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
BASH

chmod +x /tmp/apply_portal_update.sh
bash /tmp/apply_portal_update.sh
```
