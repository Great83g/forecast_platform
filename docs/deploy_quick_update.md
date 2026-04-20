# Быстрое применение обновления на сервере

Базовая последовательность (ручной режим):

```bash
cd ~/forecast_platform
git pull --rebase
source venv/bin/activate
python3 manage.py migrate
GUNICORN_PID_FILE=/run/gunicorn.pid bash deploy/restart_portal.sh
# если pid-файла нет:
# GUNICORN_PORT=8000 bash deploy/restart_portal.sh
```

## Самый простой вариант (как вы просили)

```bash
cd ~/forecast_platform
git pull --rebase
source venv/bin/activate
python3 manage.py migrate
GUNICORN_PID_FILE=/run/gunicorn.pid bash deploy/restart_portal.sh
# если pid-файла нет:
# GUNICORN_PORT=8000 bash deploy/restart_portal.sh
```

То же самое одной командой:

```bash
cd ~/forecast_platform
bash deploy/apply_portal_update.sh
```

## Автоматизированный вариант

Можно использовать готовый скрипт:

```bash
bash deploy/apply_updates.sh
```

По умолчанию скрипт:
1. Делает `git pull --rebase`.
2. Активирует `venv/bin/activate`.
3. Выполняет `python3 manage.py migrate`.
4. Перезапускает сервис через `deploy/restart_portal.sh`:
   - через `GUNICORN_PID_FILE=/run/gunicorn.pid`, если файл существует;
   - иначе через `GUNICORN_PORT=8000`.


## Частые ошибки (как на скриншоте) и быстрое исправление

Если видите ошибки:
- `cd: /path/to/forecast_platform: No such file or directory`
- `fatal: not a git repository`

значит вы выполнили команды **не в каталоге проекта** или скопировали шаблонный путь.

Используйте только рабочий путь:

```bash
cd ~/forecast_platform
pwd
ls -la .git
```

Ожидаемо:
- `pwd` показывает `/home/<user>/forecast_platform`
- `ls -la .git` не падает с ошибкой

После этого запускайте обновление:

```bash
git pull --rebase
source venv/bin/activate
python3 manage.py migrate
GUNICORN_PID_FILE=/run/gunicorn.pid bash deploy/restart_portal.sh
# если pid-файла нет:
# GUNICORN_PORT=8000 bash deploy/restart_portal.sh
```

Проверка, что всё поднялось:

```bash
curl -I http://127.0.0.1:8000/
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

## Применить конкретный коммит на портале (копипастой)

Если нужно применить **ровно один коммит** (например, который я только что дал), используйте:

```bash
cd ~/forecast_platform
bash deploy/apply_commit.sh <COMMIT_SHA>
```

Пример:

```bash
cd ~/forecast_platform
bash deploy/apply_commit.sh 96bc888
```

Что делает скрипт автоматически:
1. `git fetch --all --tags --prune`
2. `git checkout <COMMIT_SHA|branch|tag>`
3. `source venv/bin/activate`
4. `python3 manage.py migrate --plan && python3 manage.py migrate`
5. рестарт через `deploy/restart_portal.sh`
6. проверка `curl -I http://127.0.0.1:8000/`

Если нужен другой путь к проекту/venv:

```bash
PROJECT_DIR=/opt/forecast_platform VENV_PATH=/opt/venv/bin/activate \
bash deploy/apply_commit.sh <COMMIT_SHA>
```


### Если `bash deploy/apply_commit.sh <sha>` пишет `No such file or directory`

Это значит, что на сервере ещё не подтянут коммит, где добавлен этот скрипт.
Сначала выполните обычное обновление:

```bash
cd ~/forecast_platform
git pull --rebase
```

После этого скрипт появится. Если нужен просто стандартный деплой без checkout конкретного SHA — используйте:

```bash
bash deploy/apply_portal_update.sh
```


## КОПИПАСТА (ровно как нужно)

```bash
cd ~/forecast_platform
git pull --rebase
source venv/bin/activate
python3 manage.py migrate
GUNICORN_PID_FILE=/run/gunicorn.pid bash deploy/restart_portal.sh
# если pid-файла нет:
# GUNICORN_PORT=8000 bash deploy/restart_portal.sh
```

Если после `git pull --rebase` всё ещё нет `deploy/restart_portal.sh`, значит вы не в том репозитории. Проверка:

```bash
pwd
ls -la
ls -la deploy
```

Должен существовать файл `deploy/restart_portal.sh`.
