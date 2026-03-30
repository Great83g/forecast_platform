#!/usr/bin/env bash
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

# 3) обучить станцию (пример: station_id=1)
python3 manage.py train_station_models 1

# если нужно все станции:
# for sid in $(python3 manage.py shell -c "from stations.models import Station; print(' '.join(map(str, Station.objects.values_list('id', flat=True))))"); do
#   python3 manage.py train_station_models "$sid"
# done

# 4) рестарт сервиса
if [ -f /run/gunicorn.pid ]; then
  sudo env GUNICORN_PID_FILE=/run/gunicorn.pid bash deploy/restart_portal.sh
else
  sudo env GUNICORN_PORT=8000 bash deploy/restart_portal.sh
fi
