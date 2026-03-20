# Быстро применить код на сайте

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
python3 manage.py collectstatic --noinput
GUNICORN_PID_FILE=/run/gunicorn.pid bash deploy/restart_portal.sh
# если pid-файл существует, restart_portal.sh теперь сначала делает прямой HUP gunicorn
# и не лезет в systemctl, поэтому не должен спрашивать пароль администратора
# если pid-файла нет, скрипт сам попробует fallback через порт 8000
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
python3 manage.py collectstatic --noinput
GUNICORN_PID_FILE=/run/gunicorn.pid bash deploy/restart_portal.sh
# если pid-файл есть, будет прямой reload gunicorn без systemctl/password prompt
# если pid-файла нет:
# GUNICORN_PORT=8000 bash deploy/restart_portal.sh
```
