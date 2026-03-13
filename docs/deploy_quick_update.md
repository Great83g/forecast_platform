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
