#!/usr/bin/env bash
set -euo pipefail

cd "${PROJECT_DIR:-$HOME/forecast_platform}"

# shellcheck disable=SC1090
source "${VENV_PATH:-venv/bin/activate}"

REBUILD_OPERATIONAL="${REBUILD_OPERATIONAL:-1}"
FORECAST_DAYS="${FORECAST_DAYS:-7}"
export REBUILD_OPERATIONAL FORECAST_DAYS

echo "=== 1) Проверить single-axis tracker станции и их PVLIB параметры ==="
python3 manage.py shell -c "from stations.models import Station; qs=Station.objects.filter(station_kind=Station.KIND_SOLAR, mount_type=Station.MOUNT_SINGLE_AXIS_TRACKER).order_by('id'); fields=('id','name','tracker_axis_tilt','tracker_axis_azimuth','tracker_max_angle','tracker_gcr','tracker_backtrack','tracker_poa_model','tracker_albedo'); print(list(qs.values(*fields)))"

echo "=== 2) Переобучить модели только для станций с выбранным single-axis tracker ==="
python3 manage.py train_station_models --trackers-only

echo "=== 3) Пересобрать operational forecast для tracker станций ==="
if [ "$REBUILD_OPERATIONAL" = "1" ]; then
  python3 manage.py shell -c "import os; from stations.models import Station; from dashboard.services.forecast_engine import run_forecast_for_station; days=int(os.environ.get('FORECAST_DAYS','7')); qs=Station.objects.filter(station_kind=Station.KIND_SOLAR, mount_type=Station.MOUNT_SINGLE_AXIS_TRACKER).order_by('id'); [print({'station_id': st.pk, 'name': st.name, 'forecast': run_forecast_for_station(st.pk, days=days, forecast_scope='main')}) for st in qs]"
else
  echo "REBUILD_OPERATIONAL=0 — прогноз не пересобираем"
fi

echo "=== OK: tracker станции переобучены; прогноз пересобран если REBUILD_OPERATIONAL=1 ==="
