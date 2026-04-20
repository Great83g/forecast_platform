# Как применить только изменение 3 языков на странице логина

Ниже команды для сервера, чтобы **применить только один коммит** с переключателем RU/KZ/EN на странице логина.

## 1) Подготовка

```bash
cd ~/forecast_platform
source venv/bin/activate
git fetch --all --prune
```

## 2) Переключиться на боевую ветку и обновиться

> Если деплой у вас идёт из `main`:

```bash
git checkout main
git reset --hard origin/main
git clean -fd
```

## 3) Применить ТОЛЬКО нужный коммит

```bash
git cherry-pick e9a0ee7
```

Если будут конфликты:

```bash
git status
# исправить файлы
# git add <файлы>
git cherry-pick --continue
```

## 4) Нужно ли делать миграции?

**Нет, миграции не нужны** — изменён только шаблон:
- `templates/accounts/login.html`

## 5) Что выполнить после применения

```bash
python3 manage.py collectstatic --noinput
python3 manage.py cleanup_model_cache || true
sudo env GUNICORN_PID_FILE=/run/gunicorn.pid bash deploy/restart_portal.sh
```

Если PID-файла нет:

```bash
sudo env GUNICORN_PORT=8000 bash deploy/restart_portal.sh
```

## 6) Быстрая проверка

```bash
git log --oneline -n 5
```

Должен быть коммит `e9a0ee7`.

Откройте `/login/` и проверьте, что переключаются RU / KZ / EN.

## 7) Откат (если что-то не так)

```bash
git revert e9a0ee7
sudo env GUNICORN_PID_FILE=/run/gunicorn.pid bash deploy/restart_portal.sh
```

---

## Вариант «в одну вставку»

```bash
cd ~/forecast_platform && \
source venv/bin/activate && \
git fetch --all --prune && \
git checkout main && \
git reset --hard origin/main && \
git clean -fd && \
git cherry-pick e9a0ee7 && \
python3 manage.py collectstatic --noinput && \
python3 manage.py cleanup_model_cache || true && \
sudo env GUNICORN_PID_FILE=/run/gunicorn.pid bash deploy/restart_portal.sh
```
