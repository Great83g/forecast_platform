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
```

Скрипт делает полный безопасный путь:

1. `git fetch --all --tags --prune`
2. `git checkout main`
3. `git reset --hard origin/main`
4. Проверяет, что в шаблоне есть маркер нового UI: `postfactum-forecast-panel`.
   - Если маркера нет, значит `origin/main` ещё старый и PR с постфактум-панелью не попал на сервер.
5. Компилирует Python-файлы:
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
6. Запускает `python3 manage.py migrate`.
7. Запускает `python3 manage.py collectstatic --noinput`.
8. Запускает `python3 manage.py test wind.tests.WindForecastModuleTests -v 2`.
9. Перезапускает web-процесс через `deploy/restart_portal.sh`.
10. Печатает строки с `postfactum-forecast-panel` и `data-postfactum-panel-version`, чтобы можно было сразу проверить, что код действительно новый.

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
