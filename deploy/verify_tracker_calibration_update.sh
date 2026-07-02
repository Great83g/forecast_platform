#!/usr/bin/env bash
set -euo pipefail

cd "${PROJECT_DIR:-$HOME/forecast_platform}"

echo "=== 1) Где мы и какой код сейчас запущен в папке ==="
pwd
git status --short
git log --oneline -n 5

echo "=== 2) Проверка: tracker calibration код есть в checkout? ==="
rg -n "tracker_station_calibration_factor|tracker_calibrated_expected_mw|_tracker_station_calibration_curve" dashboard/services/forecast_engine.py

echo "=== 3) Проверка: команда диагностики есть? ==="
test -f dashboard/management/commands/compare_solar_forecast_modes.py
rg -n "compare_forecast_modes|FORECAST_COMPARE|forecast_weather_irr|postfact_weather_irr" \
  dashboard/management/commands/compare_solar_forecast_modes.py \
  dashboard/services/forecast_diagnostics.py

echo "=== 4) Активировать venv и проверить Django ==="
source "${VENV_PATH:-venv/bin/activate}"
python3 manage.py check

echo "=== 5) Проверка миграций ==="
python3 manage.py makemigrations --check --dry-run

echo "=== 6) Показать single-axis tracker станции в БД ==="
python3 manage.py shell -c "from stations.models import Station; qs=Station.objects.filter(station_kind=Station.KIND_SOLAR, mount_type=Station.MOUNT_SINGLE_AXIS_TRACKER).order_by('id'); print(list(qs.values('id','name','mount_type','capacity_ac_kw','capacity_dc_kw','tracker_axis_tilt','tracker_axis_azimuth','tracker_max_angle','tracker_gcr','tracker_backtrack','tracker_poa_model','tracker_albedo')[:100]))"

echo "=== 7) Диагностика operational/postfact за 25-31 мая ==="
python3 manage.py compare_solar_forecast_modes \
  --station "${DIAG_STATION:-Shu 100 MW}" \
  --from "${DIAG_FROM:-2026-05-25}" \
  --to "${DIAG_TO:-2026-05-31}" \
  --threshold "${DIAG_THRESHOLD:-5}" || true

echo "=== 8) Проверить, отвечает ли портал локально ==="
curl -fsS -I "http://127.0.0.1:${GUNICORN_PORT:-8000}/" || true

echo "=== ГОТОВО ==="
echo "Если шаги 2-5 без ошибок — код применился в checkout."
echo "Если шаг 7 печатает CSV/No forecast rows — команда диагностики работает."
echo "Если после рестарта графики старые — проверь, что gunicorn реально перезапустился и сайт смотрит на эту папку."
