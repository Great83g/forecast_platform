# Индивидуальные скрипты автоистории

Куда добавлять скрипты:
- Рекомендуемая папка: `dashboard/services/history_scripts/`

Форматы поля `Скрипт автоистории` в карточке станции:
1. Короткое имя модуля (рекомендуется):
   - `ses_8_8mw`
   - Будет импортировано как `dashboard.services.history_scripts.ses_8_8mw:build_history_dataframe`

2. Полный python путь:
   - `dashboard.services.history_scripts.ses_8_8mw:build_history_dataframe`

3. Путь к `.py` файлу:
   - `/opt/forecast/custom_history/ses_8_8mw.py:build_history_dataframe`

Функция-обработчик должна быть callable и возвращать `pandas.DataFrame`
с колонками:
- `ds`
- `irradiation`
- `air_temp`
- `pv_temp`
- `power_kw`

Сигнатура функции:

```python
def build_history_dataframe(station):
    ...
```

Дополнительно:
- В станции есть поле `Время автопроверки` — ежедневное время, когда эта станция
  должна попасть в автообновление истории.
- Если за дату есть несколько Plant Report файлов, они объединяются по часу
  (`power_kw` суммируется), затем стыкуются с метео по `ds`.

Готовый скрипт для станции 8.8 МВт:
- `ses_8_8mw` (файл `dashboard/services/history_scripts/ses_8_8mw.py`)
- Использование в поле `Скрипт автоистории`: `ses_8_8mw`
- Скрипт читает Excel вида `СЭС Кенгир 10МВт <месяц> <год>.xlsx`, где листы названы датой `dd.mm.yyyy`, и формирует почасовой DataFrame.

