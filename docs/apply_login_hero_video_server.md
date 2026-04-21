# Как применить видео на странице логина на сервере

Инструкция для уже влитого кода, где видео/постер берутся из переменных:

- `LOGIN_HERO_VIDEO_URL`
- `LOGIN_HERO_POSTER_URL`

По умолчанию используются:

- `/media/login-hero.mp4`
- `/media/login-background.png`

## 1) Залейте файлы на сервер

Самый простой способ (рекомендуется) — готовым скриптом.
Важно: сначала перейдите в **реальный путь вашего репозитория на сервере** (не обязательно `/workspace/...`):

```bash
cd /path/to/forecast_platform
```

Если после `cd` файла `deploy/apply_login_hero_media.sh` нет, значит не подтянуты свежие изменения:

```bash
git pull --rebase
```

Теперь запуск:

```bash
bash deploy/apply_login_hero_media.sh ~/Downloads/login-hero.mp4 ~/Downloads/login-background.png
```

Он сам скопирует файлы в `media/` под правильными именами и перезапустит портал.

Если хотите сделать вручную — шаги ниже.

В каталоге проекта (ручной способ):

```bash
cd /path/to/forecast_platform
cp /path/to/your/login-hero.mp4 media/login-hero.mp4
cp /path/to/your/login-background.png media/login-background.png
```

Проверьте, что файлы на месте:

```bash
ls -lh media/login-hero.mp4 media/login-background.png
```

## 2) Быстрый вариант (без смены env)

Если используете стандартные имена выше, достаточно перезапустить портал:

```bash
bash deploy/restart_portal.sh
```

или через основной update-скрипт:

```bash
bash deploy/apply_portal_update.sh
```

## 3) Вариант с кастомными путями через env

Если хотите другое имя/папку файла, задайте переменные окружения для процесса Django/Gunicorn.

Пример для текущей shell-сессии:

```bash
export LOGIN_HERO_VIDEO_URL="/media/login-hero-v2.mp4"
export LOGIN_HERO_POSTER_URL="/media/login-poster-v2.png"
bash deploy/restart_portal.sh
```

Для постоянного применения добавьте эти переменные в unit/service env (systemd/supervisor — как у вас настроено), затем перезапустите сервис.

## 4) Проверка после выката

Откройте:

- `https://intech-forecast.com/login/`

Проверьте:

1. Видео проигрывается в правой части экрана.
2. Если видео недоступно — показывается постер.
3. Ошибок 404 на видео/постер нет в DevTools Network.

## 5) Если не подхватилось

1. Очистите кэш браузера (Ctrl+F5).
2. Проверьте права на файлы в `media/`.
3. Убедитесь, что nginx/прокси отдает `/media/`.
4. Перезапустите портал повторно: `bash deploy/restart_portal.sh`.
