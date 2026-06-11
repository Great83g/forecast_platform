#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/forecast_platform}"
VENV_PATH="${VENV_PATH:-venv/bin/activate}"
HISTORY_DATE="${HISTORY_DATE:-2026-06-10}"
STATION_ID="${STATION_ID:-}"
SERVICE_NAME="${SERVICE_NAME:-forecast_portal.service}"
RESTART_SERVICE="${RESTART_SERVICE:-1}"
INSTALL_REQUIREMENTS="${INSTALL_REQUIREMENTS:-0}"

cd "$PROJECT_DIR"

echo "=== 0) Активировать venv ==="
if [ ! -f "$VENV_PATH" ]; then
  echo "ERROR: venv не найден: $PROJECT_DIR/$VENV_PATH"
  echo "Если venv в другом месте, запусти так: VENV_PATH=/path/to/venv/bin/activate bash deploy/apply_ses_1_2_history_fix.sh"
  exit 1
fi
# shellcheck disable=SC1090
source "$VENV_PATH"

echo "=== 1) Обновить main ==="
git fetch --all --tags --prune
git checkout main
git pull --ff-only origin main

echo "=== 2) Проверка коммитов ==="
git log --oneline -n 5

echo "=== 2.1) Установка зависимостей (обычно не нужна) ==="
if [ "$INSTALL_REQUIREMENTS" = "1" ] && [ -f requirements.txt ]; then
  pip install -r requirements.txt
else
  echo "Пропускаю pip install. Если надо: INSTALL_REQUIREMENTS=1 bash deploy/apply_ses_1_2_history_fix.sh"
fi

echo "=== 3) Миграции ==="
python manage.py migrate --noinput

echo "=== 4) Django check ==="
python manage.py check

echo "=== 5) Найти станцию SES/JezSolar 1.2 MW ==="
if [ -z "$STATION_ID" ]; then
  STATION_ID="$(python manage.py shell --verbosity 0 -c "from stations.models import Station; qs=Station.objects.filter(name__icontains='1.2').order_by('id'); jez=qs.filter(name__icontains='JezSolar'); qs=jez if jez.exists() else qs; print(qs.first().id if qs.exists() else '')" | tail -n 1)"
fi
if [ -z "$STATION_ID" ]; then
  echo "ERROR: не нашёл станцию 1.2 MW. Укажи явно: STATION_ID=12 bash deploy/apply_ses_1_2_history_fix.sh"
  exit 1
fi

echo "Использую STATION_ID=$STATION_ID"
python manage.py shell --verbosity 0 -c "from stations.models import Station; s=Station.objects.get(id=$STATION_ID); print(f'{s.id}: {s.name} | script={s.auto_history_script!r} | shift={s.data_shift_hours} | folder={s.auto_history_folder}')"

echo "=== 6) Включить правильный скрипт автоистории для 1.2 MW ==="
python manage.py shell --verbosity 0 -c "from stations.models import Station; s=Station.objects.get(id=$STATION_ID); s.auto_history_enabled=True; s.auto_history_script='ses_1_2mw'; s.data_shift_hours=0; s.save(update_fields=['auto_history_enabled','auto_history_script','data_shift_hours']); print(f'OK: {s.id}: {s.name} | enabled={s.auto_history_enabled} | script={s.auto_history_script!r} | shift={s.data_shift_hours}')"

echo "=== 7) Проверить импорт без изменений (dry-run) ==="
python manage.py reimport_station_history "$STATION_ID" --from-date "$HISTORY_DATE" --to-date "$HISTORY_DATE" --clear-window --dry-run

echo "=== 8) Применить: очистить день и заново импортировать историю ==="
python manage.py reimport_station_history "$STATION_ID" --from-date "$HISTORY_DATE" --to-date "$HISTORY_DATE" --clear-window

echo "=== 9) Проверить строки за $HISTORY_DATE ==="
python manage.py shell --verbosity 0 -c "from solar.models import SolarRecord; from stations.models import Station; s=Station.objects.get(id=$STATION_ID); qs=SolarRecord.objects.filter(station=s, timestamp__date='$HISTORY_DATE').order_by('timestamp'); print('rows=', qs.count()); print('---'); print('\\n'.join(f'{r.timestamp:%Y-%m-%d %H:%M} | power_kw={r.power_kw}' for r in qs))"

echo "=== 10) Рестарт сервиса ==="
if [ "$RESTART_SERVICE" = "1" ]; then
  sudo systemctl restart "$SERVICE_NAME"
  sudo systemctl status "$SERVICE_NAME" --no-pager -l | tail -n 60
else
  echo "Пропускаю рестарт. Чтобы включить: RESTART_SERVICE=1 bash deploy/apply_ses_1_2_history_fix.sh"
fi

echo "=== ГОТОВО ==="
echo "Проверь на графике 10.06.2026: значение из Plant Report на 08:00 должно быть в истории на 08:00, не на 09:00."
