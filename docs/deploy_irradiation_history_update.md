# Деплой обновления истории иррадиации GHI/POA

Этот сценарий аккуратно обновляет сервер после добавления двух новых колонок истории станции:

- `irradiation_ghi`
- `irradiation_poa`

Старое поле `irradiation` остаётся в базе и не удаляется. Миграции добавляют только nullable-поля, поэтому старые исторические записи не должны затираться.

## Команды для сервера

```bash
set -euo pipefail

cd ~/forecast_platform
LOG_FILE="deploy_irradiation_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

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

echo "=== 9) Проверка, что backend-сервис активен ==="
if sudo systemctl is-active --quiet forecast_portal.service; then
  echo "OK: forecast_portal.service active"
else
  echo "ERROR: forecast_portal.service не активен. Последние логи:"
  sudo journalctl -u forecast_portal.service -n 120 --no-pager
  exit 1
fi

echo "=== 10) Последние строки статуса сервиса (информационно, без остановки скрипта) ==="
sudo systemctl status forecast_portal.service --no-pager -l | tail -n 60 || true

echo "=== ГОТОВО ==="
echo "Лог деплоя сохранён в: $LOG_FILE"
```

## Если в конце были красные строки

Красный текст в конце не всегда означает проблему с миграциями. Например, `systemctl status` может печатать цветные строки из статуса или логов сервиса. В обновлённом сценарии выше:

- весь вывод сохраняется в файл `deploy_irradiation_YYYYMMDD_HHMMSS.log`;
- после рестарта сервис проверяется через `systemctl is-active --quiet`;
- информационный вывод `systemctl status | tail` больше не останавливает скрипт, если сам сервис уже активен.

Чтобы быстро проверить уже применённый сервер, выполните:

```bash
cd ~/forecast_platform
source venv/bin/activate

python manage.py showmigrations stations solar | tail -n 60
python manage.py shell <<'PY'
from solar.models import SolarRecord
from stations.models import Station
print([f.name for f in SolarRecord._meta.fields if "irradiation" in f.name])
print("Station.irradiation_type:", any(f.name == "irradiation_type" for f in Station._meta.fields))
PY
sudo systemctl is-active forecast_portal.service
```

Ожидаемый результат: в `showmigrations` нужные миграции отмечены `[X]`, в shell видны `irradiation`, `irradiation_ghi`, `irradiation_poa`, а сервис печатает `active`.

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
