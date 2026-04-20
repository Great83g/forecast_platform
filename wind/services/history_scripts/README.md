# Скрипты автоистории (ветер)

Папка для индивидуальных обработчиков истории ветростанций, аналогично солнечному модулю.

## Формат функции

Скрипт должен экспортировать функцию `build_history_dataframe(station)` и возвращать `pandas.DataFrame` с колонками:

- `ds` — дата/время
- `power_kw` — фактическая мощность

Дополнительно можно вернуть:

- `wind_speed_ms`
- `wind_direction_deg`
- `air_temp`
- `air_density`

## Пример значения в поле «Скрипт автоистории»

- `example_wind`
- `wind.services.history_scripts.example_wind:build_history_dataframe`
