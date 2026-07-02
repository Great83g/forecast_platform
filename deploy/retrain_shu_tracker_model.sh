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
# Django shell can print a banner such as "19 objects imported automatically" before
# our output. Print a unique marker and parse only that line, otherwise the banner
# gets captured into STATION_ID and breaks integer lookups.
STATION_ID="$(
  python3 manage.py shell -c 'import os; from stations.models import Station; name=os.environ["STATION_NAME"]; st=Station.objects.filter(name=name).first() or Station.objects.filter(name__icontains=name).first(); print("__STATION_ID__=%s" % (st.pk if st else ""))' \
    | sed -n 's/^__STATION_ID__=//p' \
    | tail -n 1
)"
if [ -z "$STATION_ID" ]; then
  echo "ERROR: station not found: $STATION_NAME" >&2
  exit 1
fi
export STATION_ID
python3 manage.py shell -c "import os; from stations.models import Station; st=Station.objects.get(pk=int(os.environ['STATION_ID'])); actual={'tracker_axis_tilt': st.tracker_axis_tilt, 'tracker_axis_azimuth': st.tracker_axis_azimuth, 'tracker_max_angle': st.tracker_max_angle, 'tracker_gcr': st.tracker_gcr, 'tracker_backtrack': st.tracker_backtrack, 'tracker_poa_model': st.tracker_poa_model, 'tracker_albedo': st.tracker_albedo}; print({'id': st.pk, 'name': st.name, 'mount_type': st.mount_type, 'AC_kW': st.capacity_ac_kw, 'DC_kW': st.capacity_dc_kw, **actual}); expected={'tracker_axis_tilt': 0.0, 'tracker_axis_azimuth': 0.0, 'tracker_max_angle': 60.0, 'tracker_gcr': 0.3105, 'tracker_backtrack': True, 'tracker_poa_model': 'perez', 'tracker_albedo': 0.2}; bad={k: (actual[k], v) for k, v in expected.items() if actual[k] != v}; print({'SHU_TRACKER_PARAM_MISMATCH': bad} if bad else {'SHU_TRACKER_PARAMS_OK': expected})"

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
