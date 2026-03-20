# Быстро применить код на сайте

## Жёсткая синхронизация с `origin/main`

Если нужно **в точности** повторить последовательность:

```bash
cd ~/forecast_platform
git fetch --all --prune
git checkout main
git reset --hard origin/main
git clean -fd
source venv/bin/activate
python3 manage.py migrate
python3 manage.py cleanup_model_cache
python3 manage.py collectstatic --noinput
sudo env GUNICORN_PID_FILE=/run/gunicorn.pid bash deploy/restart_portal.sh
# если pid-файла нет:
# sudo env GUNICORN_PORT=8000 bash deploy/restart_portal.sh
```

используй одну команду:

```bash
cd ~/forecast_platform
bash deploy/force_sync_and_restart.sh
```

Скрипт делает именно destructive sync с `origin/main`: забирает свежие refs, делает `checkout main`, `reset --hard origin/main`, `git clean -fd`, затем запускает `migrate`, `cleanup_model_cache`, `collectstatic` и рестарт через `restart_portal.sh`. Это важно, потому что обычный `git clean -fd` не удаляет ignored-файлы из `models_cache`.

## Рекомендуемый вариант — одной командой

```bash
cd ~/forecast_platform
bash deploy/apply_portal_update.sh
```

Если на сервере бывают локальные правки/временные файлы и их нужно автоматически убрать в stash перед обновлением:

```bash
cd ~/forecast_platform
STASH_LOCAL_CHANGES=1 bash deploy/apply_portal_update.sh
```

Если virtualenv лежит не в `venv/`:

```bash
cd ~/forecast_platform
VENV_PATH=/path/to/venv/bin/activate bash deploy/apply_portal_update.sh
```

## Что делает скрипт

Скрипт сам выполняет ту же последовательность команд:

```bash
cd ~/forecast_platform
git status --short
git pull --rebase
source venv/bin/activate
python3 manage.py migrate
python3 manage.py cleanup_model_cache
python3 manage.py collectstatic --noinput
GUNICORN_PID_FILE=/run/gunicorn.pid bash deploy/restart_portal.sh
# если pid-файл существует, restart_portal.sh теперь сначала делает прямой HUP gunicorn
# и не лезет в systemctl, поэтому не должен спрашивать пароль администратора
# если pid-файла нет, скрипт сам попробует fallback через порт 8000
# при ручном запуске на сервере можно использовать sudo env GUNICORN_PORT=8000 bash deploy/restart_portal.sh
```

## Полностью вручную

Если хочешь обновить совсем руками, используй именно этот порядок:

```bash
cd ~/forecast_platform
git status --short
git stash push --include-untracked -m 'manual-before-update'   # только если есть незакоммиченные локальные изменения
git pull --rebase
source venv/bin/activate
python3 manage.py migrate
python3 manage.py cleanup_model_cache
python3 manage.py collectstatic --noinput
sudo env GUNICORN_PID_FILE=/run/gunicorn.pid bash deploy/restart_portal.sh
# cleanup_model_cache нужен, потому что git clean -fd не удаляет ignored-файлы из models_cache
# если pid-файл есть, будет прямой reload gunicorn без systemctl/password prompt
# если pid-файла нет:
# sudo env GUNICORN_PORT=8000 bash deploy/restart_portal.sh
```
