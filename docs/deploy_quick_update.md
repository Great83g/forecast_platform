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

## Вариант «обновить main без обучения»

Если нужен сценарий «как в ручных командах» **без запуска обучения**, используй:

```bash
cd ~/forecast_platform
bash deploy/apply_main_update_with_training.sh
# или коротко (скрипт в корне репозитория):
bash apply_main_update_with_training.sh
```

Если получили `No such file or directory`, сначала подтяните `main`, чтобы скрипт появился:

```bash
cd ~/forecast_platform
git fetch --all --prune
git checkout main
git pull --ff-only origin main
```



## Если `git pull --rebase` пишет `cannot pull with rebase: You have unstaged changes`

Это ровно та ситуация, которая у вас на скриншоте: серверный репозиторий **грязный** и Git не даёт сделать `pull --rebase`.

### Самый быстрый безопасный вариант

```bash
cd ~/forecast_platform
git status --short
git stash push --include-untracked -m "before-portal-update"
bash deploy/apply_portal_update.sh
```

### Одной командой через обновлённый скрипт

```bash
cd ~/forecast_platform
STASH_LOCAL_CHANGES=1 bash deploy/apply_portal_update.sh
```

### Применить конкретный коммит даже при локальных изменениях

```bash
cd ~/forecast_platform
STASH_LOCAL_CHANGES=1 bash deploy/apply_commit.sh <COMMIT_SHA>
```

Например:

```bash
cd ~/forecast_platform
STASH_LOCAL_CHANGES=1 bash deploy/apply_commit.sh 15dcaa3
```

Потом можно посмотреть сохранённые stash-записи:

```bash
git stash list
```

> Важно: stash сохраняет ваши локальные серверные правки отдельно и **не мешает** подтянуть рабочую версию портала.


## Если `git pull --rebase` пишет `cannot pull with rebase: You have unstaged changes`

Это ровно та ситуация, которая у вас на скриншоте: серверный репозиторий **грязный** и Git не даёт сделать `pull --rebase`.

### Самый быстрый безопасный вариант

```bash
cd ~/forecast_platform
git status --short
git stash push --include-untracked -m "before-portal-update"
bash deploy/apply_portal_update.sh
```

### Одной командой через обновлённый скрипт

```bash
cd ~/forecast_platform
STASH_LOCAL_CHANGES=1 bash deploy/apply_portal_update.sh
```

### Применить конкретный коммит даже при локальных изменениях

```bash
cd ~/forecast_platform
STASH_LOCAL_CHANGES=1 bash deploy/apply_commit.sh <COMMIT_SHA>
```

Например:

```bash
cd ~/forecast_platform
STASH_LOCAL_CHANGES=1 bash deploy/apply_commit.sh 15dcaa3
```

Потом можно посмотреть сохранённые stash-записи:

```bash
git stash list
```

> Важно: stash сохраняет ваши локальные серверные правки отдельно и **не мешает** подтянуть рабочую версию портала.


## Если `git pull --rebase` пишет `cannot pull with rebase: You have unstaged changes`

Это ровно та ситуация, которая у вас на скриншоте: серверный репозиторий **грязный** и Git не даёт сделать `pull --rebase`.

### Самый быстрый безопасный вариант

```bash
cd ~/forecast_platform
git status --short
git stash push --include-untracked -m "before-portal-update"
bash deploy/apply_portal_update.sh
```

### Одной командой через обновлённый скрипт

```bash
cd ~/forecast_platform
STASH_LOCAL_CHANGES=1 bash deploy/apply_portal_update.sh
```

### Применить конкретный коммит даже при локальных изменениях

```bash
cd ~/forecast_platform
STASH_LOCAL_CHANGES=1 bash deploy/apply_commit.sh <COMMIT_SHA>
```

Например:

```bash
cd ~/forecast_platform
STASH_LOCAL_CHANGES=1 bash deploy/apply_commit.sh 15dcaa3
```

Потом можно посмотреть сохранённые stash-записи:

```bash
git stash list
```

> Важно: stash сохраняет ваши локальные серверные правки отдельно и **не мешает** подтянуть рабочую версию портала.


## Если `git pull --rebase` пишет `cannot pull with rebase: You have unstaged changes`

Это ровно та ситуация, которая у вас на скриншоте: серверный репозиторий **грязный** и Git не даёт сделать `pull --rebase`.

### Самый быстрый безопасный вариант

```bash
cd ~/forecast_platform
git status --short
git stash push --include-untracked -m "before-portal-update"
bash deploy/apply_portal_update.sh
```

### Одной командой через обновлённый скрипт

```bash
cd ~/forecast_platform
STASH_LOCAL_CHANGES=1 bash deploy/apply_portal_update.sh
```

### Применить конкретный коммит даже при локальных изменениях

```bash
cd ~/forecast_platform
STASH_LOCAL_CHANGES=1 bash deploy/apply_commit.sh <COMMIT_SHA>
```

Например:

```bash
cd ~/forecast_platform
STASH_LOCAL_CHANGES=1 bash deploy/apply_commit.sh 15dcaa3
```

Потом можно посмотреть сохранённые stash-записи:

```bash
git stash list
```

> Важно: stash сохраняет ваши локальные серверные правки отдельно и **не мешает** подтянуть рабочую версию портала.


## Если `git pull --rebase` пишет `cannot pull with rebase: You have unstaged changes`

Это ровно та ситуация, которая у вас на скриншоте: серверный репозиторий **грязный** и Git не даёт сделать `pull --rebase`.

### Самый быстрый безопасный вариант

```bash
cd ~/forecast_platform
git status --short
git stash push --include-untracked -m "before-portal-update"
bash deploy/apply_portal_update.sh
```

### Одной командой через обновлённый скрипт

```bash
cd ~/forecast_platform
STASH_LOCAL_CHANGES=1 bash deploy/apply_portal_update.sh
```

### Применить конкретный коммит даже при локальных изменениях

```bash
cd ~/forecast_platform
STASH_LOCAL_CHANGES=1 bash deploy/apply_commit.sh <COMMIT_SHA>
```

Например:

```bash
cd ~/forecast_platform
STASH_LOCAL_CHANGES=1 bash deploy/apply_commit.sh 15dcaa3
```

Потом можно посмотреть сохранённые stash-записи:

```bash
git stash list
```

> Важно: stash сохраняет ваши локальные серверные правки отдельно и **не мешает** подтянуть рабочую версию портала.


## Если `git pull --rebase` пишет `cannot pull with rebase: You have unstaged changes`

Это ровно та ситуация, которая у вас на скриншоте: серверный репозиторий **грязный** и Git не даёт сделать `pull --rebase`.

### Самый быстрый безопасный вариант

```bash
cd ~/forecast_platform
git status --short
git stash push --include-untracked -m "before-portal-update"
bash deploy/apply_portal_update.sh
```

### Одной командой через обновлённый скрипт

```bash
cd ~/forecast_platform
STASH_LOCAL_CHANGES=1 bash deploy/apply_portal_update.sh
```

### Применить конкретный коммит даже при локальных изменениях

```bash
cd ~/forecast_platform
STASH_LOCAL_CHANGES=1 bash deploy/apply_commit.sh <COMMIT_SHA>
```

Например:

```bash
cd ~/forecast_platform
STASH_LOCAL_CHANGES=1 bash deploy/apply_commit.sh 15dcaa3
```

Потом можно посмотреть сохранённые stash-записи:

```bash
git stash list
```

> Важно: stash сохраняет ваши локальные серверные правки отдельно и **не мешает** подтянуть рабочую версию портала.


## Если `git pull --rebase` пишет `cannot pull with rebase: You have unstaged changes`

Это ровно та ситуация, которая у вас на скриншоте: серверный репозиторий **грязный** и Git не даёт сделать `pull --rebase`.

### Самый быстрый безопасный вариант

```bash
cd ~/forecast_platform
git status --short
git stash push --include-untracked -m "before-portal-update"
bash deploy/apply_portal_update.sh
```

### Одной командой через обновлённый скрипт

```bash
cd ~/forecast_platform
STASH_LOCAL_CHANGES=1 bash deploy/apply_portal_update.sh
```

### Применить конкретный коммит даже при локальных изменениях

```bash
cd ~/forecast_platform
STASH_LOCAL_CHANGES=1 bash deploy/apply_commit.sh <COMMIT_SHA>
```

Например:

```bash
cd ~/forecast_platform
STASH_LOCAL_CHANGES=1 bash deploy/apply_commit.sh 15dcaa3
```

Потом можно посмотреть сохранённые stash-записи:

```bash
git stash list
```

> Важно: stash сохраняет ваши локальные серверные правки отдельно и **не мешает** подтянуть рабочую версию портала.


## Если `git pull --rebase` пишет `cannot pull with rebase: You have unstaged changes`

Это ровно та ситуация, которая у вас на скриншоте: серверный репозиторий **грязный** и Git не даёт сделать `pull --rebase`.

### Самый быстрый безопасный вариант

```bash
cd ~/forecast_platform
git status --short
git stash push --include-untracked -m "before-portal-update"
bash deploy/apply_portal_update.sh
```

### Одной командой через обновлённый скрипт

```bash
cd ~/forecast_platform
STASH_LOCAL_CHANGES=1 bash deploy/apply_portal_update.sh
```

### Применить конкретный коммит даже при локальных изменениях

```bash
cd ~/forecast_platform
STASH_LOCAL_CHANGES=1 bash deploy/apply_commit.sh <COMMIT_SHA>
```

Например:

```bash
cd ~/forecast_platform
STASH_LOCAL_CHANGES=1 bash deploy/apply_commit.sh 15dcaa3
```

Потом можно посмотреть сохранённые stash-записи:

```bash
git stash list
```

> Важно: stash сохраняет ваши локальные серверные правки отдельно и **не мешает** подтянуть рабочую версию портала.


## Если `git pull --rebase` пишет `cannot pull with rebase: You have unstaged changes`

Это ровно та ситуация, которая у вас на скриншоте: серверный репозиторий **грязный** и Git не даёт сделать `pull --rebase`.

### Самый быстрый безопасный вариант

```bash
cd ~/forecast_platform
git status --short
git stash push --include-untracked -m "before-portal-update"
bash deploy/apply_portal_update.sh
```

### Одной командой через обновлённый скрипт

```bash
cd ~/forecast_platform
STASH_LOCAL_CHANGES=1 bash deploy/apply_portal_update.sh
```

### Применить конкретный коммит даже при локальных изменениях

```bash
cd ~/forecast_platform
STASH_LOCAL_CHANGES=1 bash deploy/apply_commit.sh <COMMIT_SHA>
```

Например:

```bash
cd ~/forecast_platform
STASH_LOCAL_CHANGES=1 bash deploy/apply_commit.sh 15dcaa3
```

Потом можно посмотреть сохранённые stash-записи:

```bash
git stash list
```

> Важно: stash сохраняет ваши локальные серверные правки отдельно и **не мешает** подтянуть рабочую версию портала.

## Применить конкретный коммит на портале (копипастой)

Если нужно применить **ровно один коммит** (например, который я только что дал), используйте:

```bash
# обычный запуск
bash deploy/apply_main_update_with_training.sh
```

Скрипт не запускает обучение: шаг `# 3` оставлен пустым специально.

### Если нужно просто вставить команды на сервере (без скрипта)

```bash
cd ~/forecast_platform
source venv/bin/activate
set -euo pipefail

# 1) подтянуть код
git fetch --all --prune
git checkout main
git pull --ff-only origin main

# 2) применить django-шаги
python3 manage.py migrate
python3 manage.py cleanup_model_cache
python3 manage.py collectstatic --noinput

# 3) БЕЗ обучения (ничего не запускаем)

# 4) рестарт сервиса
if [ -f /run/gunicorn.pid ]; then
  sudo env GUNICORN_PID_FILE=/run/gunicorn.pid bash deploy/restart_portal.sh
else
  sudo env GUNICORN_PORT=8000 bash deploy/restart_portal.sh
fi
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
