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
