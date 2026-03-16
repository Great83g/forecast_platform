# PR #318 — что нажимать в конфликтах (Current / Incoming)

PR: `XGB: scale predictions, train on y_over_expected... #318`

Конфликтующие файлы:
- `dashboard/services/forecast_engine.py`
- `dashboard/services/train_models.py`
- `dashboard/tests.py`

## Коротко: везде выбирай **Accept current change**

Причина: в `current` лежат последние правки с калибровкой `xgb_calib_mult` и тестами под неё.
`incoming` обычно старее и откатывает часть фиксов.

---

## По твоим скринам (1..5)

### 1) `forecast_engine.py` — блок `calib_mult = ...`
**Нажать:** `Accept current change`

Оставить:
```py
calib_mult = (xgb_meta or {}).get("xgb_calib_mult", 1.0)
...
calib_mult = float(np.clip(calib_mult, 0.25, 12.0))
```

### 2) `forecast_engine.py` — `return out * expected * calib_mult`
**Нажать:** `Accept current change`

### 3) `forecast_engine.py` — `return out * cap_used * calib_mult`
**Нажать:** `Accept current change`

### 4) `train_models.py` — пост-калибровка после fit (`pred_train`, `ratio`, `calib_mult`)
**Нажать:** `Accept current change`

### 5) `train_models.py` — meta поле `"xgb_calib_mult": calib_mult`
**Нажать:** `Accept current change`

### `tests.py` (скрин с одним конфликтом)
Тест `test_applies_xgb_calibration_multiplier` должен остаться.
**Нажать:** `Accept current change`

---

## После выбора

1. Нажми `Mark as resolved` в каждом файле.
2. Нажми `Commit merge`.
3. Быстрая проверка (локально):

```bash
rg -n "^(<<<<<<<|=======|>>>>>>>)" dashboard/services/forecast_engine.py dashboard/services/train_models.py dashboard/tests.py
```

Должно вернуть пусто.

---

## Если хочешь решить локально (надёжнее)

```bash
cd ~/forecast_platform
git fetch origin
git checkout <ветка_PR_318>
git merge origin/main
# руками оставить current-варианты как выше
git add dashboard/services/forecast_engine.py dashboard/services/train_models.py dashboard/tests.py
git commit -m "Resolve PR #318 conflicts keeping XGB calibration path"
git push
```
