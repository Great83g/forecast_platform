# Применение tracker calibration на сервере

Скопируй весь блок ниже и выполни на сервере. Он сначала обновит `main`, затем запустит готовый deploy-скрипт с проверками, миграциями, статикой, рестартом и диагностикой `Shu 100 MW` за `2026-05-25`–`2026-05-31`.

```bash
#!/usr/bin/env bash
set -euo pipefail

cd ~/forecast_platform

echo "=== 0) Активировать venv ==="
source venv/bin/activate

echo "=== 1) Обновить main ==="
git fetch --all --tags --prune
git checkout main
git pull --ff-only origin main

echo "=== 2) Проверка коммитов ==="
git log --oneline -n 5

echo "=== 3) Применить tracker calibration update ==="
RUN_TESTS="${RUN_TESTS:-1}" \
DIAG_STATION="${DIAG_STATION:-Shu 100 MW}" \
DIAG_FROM="${DIAG_FROM:-2026-05-25}" \
DIAG_TO="${DIAG_TO:-2026-05-31}" \
DIAG_THRESHOLD="${DIAG_THRESHOLD:-5}" \
bash deploy/apply_tracker_calibration_simple.sh
```

Если на сервере нужно быстрее и без тестов:

```bash
cd ~/forecast_platform
git pull --ff-only origin main
RUN_TESTS=0 bash deploy/apply_tracker_calibration_simple.sh
```

После выполнения проверь, что в конце есть строка `=== OK ===` и напечатан short SHA текущего коммита.
