# Деплой обновления истории иррадиации GHI/POA

Этот сценарий аккуратно обновляет сервер после добавления двух новых колонок истории станции:

- `irradiation_ghi`
- `irradiation_poa`

Старое поле `irradiation` остаётся в базе и не удаляется. Миграции добавляют только nullable-поля, поэтому старые исторические записи не должны затираться.

## Команды для сервера

```bash
set -euo pipefail

cd ~/forecast_platform

echo "=== 0) Активировать venv ==="
source venv/bin/activate

echo "=== 1) Обновить main ==="
git fetch --all --tags --prune
git checkout main
git pull --ff-only origin main

echo "=== 2) Проверка последних коммитов ==="
git log --oneline -n 5

echo "=== 2.1) Установка зависимостей (если есть requirements.txt) ==="
if [ -f requirements.txt ]; then
  pip install -r requirements.txt
else
  echo "requirements.txt не найден — пропускаю pip install"
fi

echo "=== 3) Проверка плана миграций ==="
python manage.py showmigrations stations solar | tail -n 40

echo "=== 4) Миграции ==="
python manage.py migrate --noinput

echo "=== 5) Статика ==="
python manage.py collectstatic --noinput

echo "=== 6) Django check ==="
python manage.py check

echo "=== 7) Быстрая проверка новых полей в БД через Django shell ==="
python manage.py shell <<'PY'
from solar.models import SolarRecord
from stations.models import Station

record_fields = {field.name for field in SolarRecord._meta.fields}
station_fields = {field.name for field in Station._meta.fields}

required_record_fields = {"irradiation", "irradiation_ghi", "irradiation_poa"}
required_station_fields = {"irradiation_type"}

missing_record = required_record_fields - record_fields
missing_station = required_station_fields - station_fields

if missing_record or missing_station:
    raise SystemExit(f"Missing fields: SolarRecord={missing_record}, Station={missing_station}")

print("OK: поля SolarRecord.irradiation/irradiation_ghi/irradiation_poa и Station.irradiation_type доступны")
PY

echo "=== 8) Рестарт backend-сервиса ==="
sudo systemctl restart forecast_portal.service

echo "=== 9) Статус backend-сервиса ==="
sudo systemctl status forecast_portal.service --no-pager -l | tail -n 60
```

## После деплоя

1. Откройте карточку станции и проверьте настройку **Тип старой irradiation**:
   - `GHI`, если старое поле `irradiation` было горизонтальной радиацией.
   - `POA`, если старое поле `irradiation` было радиацией в плоскости панелей.
2. Для одноосевых трекеров загружайте `irradiation_ghi` / `GHI` обязательно, если есть такая колонка. `irradiation_poa` / `POA` можно загружать отдельно для диагностики, обучения и валидации.
3. При загрузке истории можно явно заполнить поля маппинга:
   - `GHI column`
   - `POA column`
   - `Power column`
   - `Air temp column`
   - `PV temp column`

## Важно

- Скрипт не очищает историю.
- `python manage.py migrate --noinput` применяет миграции, которые добавляют новые nullable-поля.
- Старое поле `irradiation` сохраняется для обратной совместимости и экспорта старых данных.
