# PR #317: как быстро закрыть конфликты (Current / Incoming)

Для PR `XGB: training & prediction scaling, low-confidence... #317` конфликтуют:
- `dashboard/services/forecast_engine.py`
- `dashboard/tests.py`

## Вариант 1: прямо в GitHub conflict editor

### Файл `dashboard/services/forecast_engine.py`

#### Конфликт №1 (функция `_postprocess_xgb_prediction`)
**Выбирай: `Accept current change`**.

Нужна версия с сигнатурой:
- `df_feat: Optional[pd.DataFrame] = None`

и поддержкой двух target:
- `y_over_expected`
- `y_permw`

Не оставляй старую incoming-версию только с `y_permw`.

#### Конфликт №2 (блок `is_over_expected` внутри `_postprocess_xgb_prediction`)
**Выбирай: `Accept current change`**.

Нужен расчёт:
- floor (`y_expected_floor_mw`),
- expected из `Irradiation`,
- `return out * expected`.

#### Конфликт №3 (функция `_xgb_is_systematically_low`)
**Выбирай: `Accept current change`**.

Эта функция должна остаться, чтобы XGB не тянул итог вниз при явно заниженном сигнале.

#### Конфликт №4 (вызов `_postprocess_xgb_prediction`)
Оставь вызов с `df_feat=feat`:

```python
y_xgb = _postprocess_xgb_prediction(y_xgb, xgb_meta, capacity_mw=capacity_mw, df_feat=feat)
```

---

### Файл `dashboard/tests.py`

#### Конфликт №1 (import block)
Нужно оставить **оба импорта**:
- `_xgb_is_systematically_low`
- `_prepare_xgb_training_frame`

Если есть кнопка, можно `Accept both changes`, потом руками удалить дубли и маркеры.

#### Конфликт №2 (тест `test_converts_over_expected_predictions_to_mw`)
**Оставь этот тест (current)**, он проверяет новую ветку `y_over_expected`.

---

После выбора вариантов нажми `Mark as resolved` и `Commit merge`.

## Вариант 2: локально (быстрее и надёжнее)

```bash
cd ~/forecast_platform
git fetch origin
git checkout qn5w5v-codex/fix-prediction-accuracy-issue
git merge origin/main
```

Дальше вручную правишь 2 файла по правилам выше, затем:

```bash
git add dashboard/services/forecast_engine.py dashboard/tests.py
git commit -m "Resolve PR #317 conflicts keeping current XGB scaling + low-confidence + tests"
git push
```

## Быстрая проверка, что маркеров не осталось

```bash
rg -n "^(<<<<<<<|=======|>>>>>>>)" dashboard/services/forecast_engine.py dashboard/tests.py
```

Команда должна вернуть пустой результат.
