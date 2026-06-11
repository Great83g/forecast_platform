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
- `irradiation` (legacy, для обратной совместимости)
- `irradiation_ghi` (опционально, GHI)
- `irradiation_poa` (опционально, POA)
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
- Использование в поле `Скрипт автоистории`: `ses_8_8mw` (также сработают `ses 8 8mw`, `/history_scripts/ses_8_8mw`, `history_scripts\ses_8_8mw`, `dashboard/services/history_scripts/ses_8_8mw`)
- Скрипт читает Excel вида `СЭС Кенгир 10МВт <месяц> <год>.xlsx`, где листы названы датой `dd.mm.yyyy`, и формирует почасовой DataFrame.

Готовый скрипт для станции 1.2 МВт (загрузка из стандартного файла портала):
- `ses_1_2mw` (файл `dashboard/services/history_scripts/ses_1_2mw.py`)
- Использование в поле `Скрипт автоистории`: `ses_1_2mw`
- Для пары `D222152*.csv.gz` + `Plant Report_SPP JezSolar 1.2 MW_*.xlsx` скрипт сохраняет часы как в исходных файлах без дополнительного `+1ч`: строка отчёта `08:00` записывается в историю на `08:00`.
- Скрипт также читает стандартные `.csv`/`.xlsx` из папки автоимпорта и ожидает колонки: `ds, Irradiation/Irradiation_GHI/Irradiation_POA, Air_Temp, PV_Temp, Power_KW` (допускается `timestamp` вместо `ds`).
- Код для поля «Скрипт автоистории» на сервере: `ses_1_2mw`
- После деплоя исправления можно принудительно пересобрать день из файлов на сервере:

```bash
python manage.py reimport_station_history <station_id> --from-date 2026-06-10 --to-date 2026-06-10 --clear-window
```

- Готовый bash-скрипт для применения на Ubuntu-сервере:

```bash
bash deploy/apply_ses_1_2_history_fix.sh
```

Если автоопределение станции не подходит, укажите id вручную:

```bash
STATION_ID=<station_id> HISTORY_DATE=2026-06-10 bash deploy/apply_ses_1_2_history_fix.sh
```


Готовый скрипт для станции СЭС Балхаш (станция 50):
- `ses_50_balkhash` (файл `dashboard/services/history_scripts/ses_50_balkhash.py`)
- Использование в поле `Скрипт автоистории`: `ses_50_balkhash`
- Скрипт рекурсивно читает Excel (`.xlsx/.xlsm/.xltx/.xltm`) в папке автоимпорта (включая подпапки `24/25/26`), ищет заголовки `Время`, `Мощность актив...`, `Иррадиация`, `Температура воздуха`, `Температура ФЭМ`, агрегирует 15-минутные строки в час и сохраняет `power_kw = sum(power_raw) * 1000`.


Готовый скрипт для станции СЭС Шиели 20 МВт:
- `ses_shieli_20mw` (файл `dashboard/services/history_scripts/ses_shieli_20mw.py`)
- Использование в поле `Скрипт автоистории`: `ses_shieli_20mw`
- Скрипт читает Excel-суточные отчёты (дата/время в колонке A, генерация в колонке C), применяет базовый сдвиг `+4ч` (плюс `Сдвиг данных` станции), удаляет ночной шум и возвращает почасовую историю в формате автоистории.
- Код для поля «Скрипт автоистории» на сервере (рекомендуется): `ses_shieli_20mw`
- Полный вариант для сервера: `dashboard.services.history_scripts.ses_shieli_20mw:build_history_dataframe`
- Дополнительный алиас для варианта написания «Шеле»: `ses_shele_20mw`
