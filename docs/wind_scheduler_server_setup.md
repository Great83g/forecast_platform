# Настройка автопрогноза ветра на сервере

Если автопрогноз «не идёт», чаще всего не запущен планировщик `run_scheduled_forecasts`.

## Быстрый запуск вручную

```bash
cd /path/to/forecast_platform
bash deploy/run_scheduler_tick.sh
```

## Рекомендуемый cron (каждые 10 минут)

```bash
*/10 * * * * cd /path/to/forecast_platform && /bin/bash deploy/run_scheduler_tick.sh >> /var/log/forecast_scheduler.log 2>&1
```

## Что проверять

1. У станции включён «Авто‑прогноз».
2. Указано время запуска (по серверному timezone).
3. Ветка ветра теперь строит прогноз и без email.
4. Email отправляется только если заполнены получатели.
