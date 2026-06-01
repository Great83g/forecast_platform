#!/usr/bin/env bash
set -euo pipefail

cd "${PROJECT_DIR:-$HOME/forecast_platform}"

echo "=== 0) Активировать venv ==="
source "${VENV_PATH:-venv/bin/activate}"

STATION_NAME="${STATION_NAME:-Shu 100 MW}"
POSTFACT_FROM="${POSTFACT_FROM:-2026-05-25}"
POSTFACT_TO="${POSTFACT_TO:-2026-05-31}"
REBUILD_OPERATIONAL="${REBUILD_OPERATIONAL:-1}"
REBUILD_POSTFACT_TEST="${REBUILD_POSTFACT_TEST:-1}"

export STATION_NAME POSTFACT_FROM POSTFACT_TO REBUILD_OPERATIONAL REBUILD_POSTFACT_TEST

echo "=== 1) Найти станцию ==="
STATION_ID="$(python3 manage.py shell -c "import os; from stations.models import Station; name=os.environ['STATION_NAME']; st=Station.objects.filter(name=name).first() or Station.objects.filter(name__icontains=name).first(); print(st.pk if st else '')")"
if [ -z "$STATION_ID" ]; then
  echo "ERROR: station not found: $STATION_NAME" >&2
  exit 1
fi
export STATION_ID
python3 manage.py shell -c "import os; from stations.models import Station; st=Station.objects.get(pk=os.environ['STATION_ID']); print({'id': st.pk, 'name': st.name, 'mount_type': st.mount_type, 'AC_kW': st.capacity_ac_kw, 'DC_kW': st.capacity_dc_kw})"

echo "=== 2) Переобучить модель станции ==="
python3 manage.py train_station_models "$STATION_ID"

echo "=== 3) Пересобрать operational forecast (main, будущий горизонт) ==="
if [ "$REBUILD_OPERATIONAL" = "1" ]; then
  python3 manage.py shell -c "import os; from dashboard.services.forecast_engine import run_forecast_for_station; res=run_forecast_for_station(int(os.environ['STATION_ID']), days=7, forecast_scope='main'); print(res)"
else
  echo "REBUILD_OPERATIONAL=0 — пропущено"
fi

echo "=== 4) Пересобрать postfact test forecast за период для диагностики ==="
if [ "$REBUILD_POSTFACT_TEST" = "1" ]; then
  python3 manage.py shell -c "import os; from datetime import date, timedelta; from dashboard.services.forecast_engine import run_forecast_for_station; start=date.fromisoformat(os.environ['POSTFACT_FROM']); end=date.fromisoformat(os.environ['POSTFACT_TO']); dates=[start + timedelta(days=i) for i in range((end-start).days + 1)]; res=run_forecast_for_station(int(os.environ['STATION_ID']), days=len(dates), forecast_scope='test', target_dates=dates); print(res)"
else
  echo "REBUILD_POSTFACT_TEST=0 — пропущено"
fi

echo "=== 5) Диагностика operational/postfact ==="
python3 manage.py compare_solar_forecast_modes \
  --station-id "$STATION_ID" \
  --from "$POSTFACT_FROM" \
  --to "$POSTFACT_TO" \
  --threshold "${DIAG_THRESHOLD:-5}" || true

echo "=== 6) Перезапуск портала ==="
if [ -f "${GUNICORN_PID_FILE:-/run/gunicorn.pid}" ]; then
  sudo env GUNICORN_PID_FILE="${GUNICORN_PID_FILE:-/run/gunicorn.pid}" bash deploy/restart_portal.sh
else
  sudo env GUNICORN_PORT="${GUNICORN_PORT:-8000}" PREFER_DIRECT_GUNICORN_RELOAD=1 bash deploy/restart_portal.sh
fi

echo "=== OK: модель переобучена, прогнозы пересобраны где включено ==="
