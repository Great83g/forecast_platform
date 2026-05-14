# Деплой правок WindForecastRun / постфактум-прогноза

## Копируй на сервер сразу

```bash
cd ~/forecast_platform
source venv/bin/activate
set -euo pipefail

git fetch --all --tags --prune
git checkout main
git reset --hard origin/main
git log -1 --oneline

rg -n "postfactum-forecast-panel|data-postfactum-panel-version" wind/templates/wind/station_forecast_list.html

python3 -m py_compile \
  manage.py \
  backend/urls.py \
  dashboard/services/open_meteo.py \
  dashboard/services/vc_weather.py \
  wind/models.py \
  wind/views.py \
  wind/services/forecast_runs.py \
  wind/services/forecasting.py \
  dashboard/services/forecast_scheduler.py \
  wind/tests.py
python3 -m compileall -q wind dashboard/services

python3 manage.py migrate
python3 manage.py collectstatic --noinput
python3 manage.py test wind.tests.WindForecastModuleTests -v 2

if [ -f /run/gunicorn.pid ]; then
  sudo env GUNICORN_PID_FILE=/run/gunicorn.pid bash deploy/restart_portal.sh
else
  sudo env GUNICORN_PORT=8000 PREFER_DIRECT_GUNICORN_RELOAD=1 bash deploy/restart_portal.sh
fi
```

## Короткий вариант через скрипт

```bash
cd ~/forecast_platform
source venv/bin/activate
bash deploy/apply_wind_forecast_runs_update.sh
# если по привычке набрал без h, тоже работает:
# bash deploy/apply_wind_forecast_runs_update.s
```

Да, короткий вариант ниже можно запускать на сервере повторно:

```bash
cd ~/forecast_platform
source venv/bin/activate
bash deploy/apply_wind_forecast_runs_update.sh
```

Он не обучает модели и не удаляет прогнозы. Основные изменения, которые он делает: синхронизирует код с `origin/main`, применяет Django-миграции, собирает статику, прогоняет тесты и перезапускает web-процесс. Перед потенциально рискованными шагами скрипт сохраняет состояние git и делает резервную копию SQLite-базы в `backups/wind_forecast_runs/`.

Скрипт делает полный безопасный путь:

1. Сохраняет текущие `git status`, обычный diff и staged diff в `backups/wind_forecast_runs/`.
2. `git fetch --all --tags --prune`
3. `git checkout main`
4. `git reset --hard origin/main`
5. Проверяет, что в шаблоне есть маркер нового UI: `postfactum-forecast-panel`.
   - Если маркера нет, значит `origin/main` ещё старый и PR с постфактум-панелью не попал на сервер.
6. Компилирует Python-файлы:
   - `manage.py`
   - `backend/urls.py`
   - `dashboard/services/open_meteo.py`
   - `dashboard/services/vc_weather.py`
   - `wind/models.py`
   - `wind/views.py`
   - `wind/services/forecast_runs.py`
   - `wind/services/forecasting.py`
   - `dashboard/services/forecast_scheduler.py`
   - `wind/tests.py`
7. Проверяет, что нет забытых миграций: `python3 manage.py makemigrations --check --dry-run`.
8. Если база стандартная SQLite (`db.sqlite3`), делает резервную копию перед миграцией.
9. Показывает план миграций: `python3 manage.py migrate --plan`.
10. Запускает `python3 manage.py migrate`. Повторный запуск безопасен: уже применённые миграции Django пропускает.
11. Запускает `python3 manage.py collectstatic --noinput`.
12. Запускает `python3 manage.py test wind.tests.WindForecastModuleTests -v 2`.
13. Перезапускает web-процесс через `deploy/restart_portal.sh`.
14. Печатает строки с `postfactum-forecast-panel` и `data-postfactum-panel-version`, чтобы можно было сразу проверить, что код действительно новый.

## Что делать, если переживаешь за сервер

- Если на сервере есть локальные незакоммиченные правки, скрипт сохранит их diff в `backups/wind_forecast_runs/`, но `git reset --hard origin/main` всё равно приведёт tracked-файлы к состоянию `origin/main`. Поэтому перед запуском можно вручную проверить `git status --short`.
- Если нужно применить шаги к текущему checkout без `git fetch`/`git reset --hard`, запусти:

```bash
SKIP_SYNC=1 bash deploy/apply_wind_forecast_runs_update.sh
```

- Если тесты на проде слишком долго идут, можно временно пропустить только тестовый шаг:

```bash
RUN_TESTS=0 bash deploy/apply_wind_forecast_runs_update.sh
```

- Резервные копии SQLite создаются в `backups/wind_forecast_runs/db_YYYYMMDD_HHMMSS.sqlite3`.

## Ручной вариант

```bash
cd ~/forecast_platform
source venv/bin/activate
set -euo pipefail

git fetch --all --tags --prune
git checkout main
git reset --hard origin/main
git log -1 --oneline

rg -n "postfactum-forecast-panel|data-postfactum-panel-version" wind/templates/wind/station_forecast_list.html

python3 -m py_compile \
  manage.py \
  backend/urls.py \
  dashboard/services/open_meteo.py \
  dashboard/services/vc_weather.py \
  wind/models.py \
  wind/views.py \
  wind/services/forecast_runs.py \
  wind/services/forecasting.py \
  dashboard/services/forecast_scheduler.py \
  wind/tests.py
python3 -m compileall -q wind dashboard/services

python3 manage.py migrate
python3 manage.py collectstatic --noinput
python3 manage.py test wind.tests.WindForecastModuleTests -v 2

if [ -f /run/gunicorn.pid ]; then
  sudo env GUNICORN_PID_FILE=/run/gunicorn.pid bash deploy/restart_portal.sh
else
  sudo env GUNICORN_PORT=8000 PREFER_DIRECT_GUNICORN_RELOAD=1 bash deploy/restart_portal.sh
fi
```

После этого открой страницу прогноза ветра и сделай hard refresh: `Ctrl+F5`.

## Важно

Если после `git reset --hard origin/main` в выводе `git log -1 --oneline` старый merge commit, а `rg postfactum-forecast-panel ...` ничего не находит — сервер обновлён правильно, но **в origin/main ещё нет нужного PR**. В таком случае страница не изменится, пока PR не будет влит в `main`.
