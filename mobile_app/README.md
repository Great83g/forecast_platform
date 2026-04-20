# Forecast Mobile App (Flutter)

Базовый каркас Flutter-приложения добавлен в репозиторий.

## Локальный запуск

1. Установите Flutter SDK (stable) и Android SDK.
2. В каталоге `mobile_app` выполните:

```bash
flutter pub get
flutter run
```

## Подключение к backend (шаг 1)

Сейчас логин отправляет запрос `POST` в endpoint из `LOGIN_PATH` (по умолчанию `/auth/login`).

По умолчанию используется URL:

- `http://10.0.2.2:8000` (для Android-эмулятора, если backend запущен на том же ПК)

Если backend на другом сервере, запускайте так:

```bash
flutter run --dart-define=API_BASE_URL=http://<SERVER_IP>:<PORT>
```

Пример:

```bash
flutter run --dart-define=API_BASE_URL=http://192.168.1.50:8000
```

Если endpoint логина другой, укажи его явно:

```bash
flutter run --dart-define=API_BASE_URL=https://intech-forecast.com --dart-define=LOGIN_PATH=/users/login
```


## Если на эмуляторе появляется `System UI isn't responding`

Это проблема эмулятора/ресурсов ПК, а не кода экрана логина.

1. Используйте **стабильный образ** Android (API 36, не Preview).
2. В Device Manager сделайте **Cold Boot Now**.
3. Если не помогло — **Wipe Data** у виртуального устройства.
4. Перезапустите ADB и проверьте, что статус `device`, а не `offline`:

```bash
adb kill-server
adb start-server
adb devices
```

5. Запустите приложение явно на эмуляторе:

```bash
flutter devices
flutter run -d <emulator_id>
```

6. Если сборка зависала после обновления SDK/NDK:

```bash
flutter clean
flutter pub get
```

### Рекомендуемая конфигурация AVD для слабого ПК

- Device: Pixel 6 / Pixel 7
- System image: Google Play x86_64 (stable)
- Graphics: Software или Compatibility
- RAM: 2048–3072 MB
- Не запускать несколько эмуляторов одновременно

## Следующие шаги

- Сохранение access/refresh token в secure storage.
- Авто-обновление access token (refresh flow).
- Настройка окружений `dev/stage/prod`.
- Добавить unit/widget тесты (TDD).

## Что уже реализовано в MVP

- Экран логина на Flutter (`lib/main.dart`) с вводом email/username.
- Валидация email/пароля.
- Состояние отправки формы с индикатором загрузки.
- Реальный HTTP-вызов login через Dio с настраиваемым `LOGIN_PATH`.
- Отображение текущего `baseUrl` на экране для быстрой диагностики.


## Шаг 2 (сделано)

- После успешного login приложение открывает `HomeScreen`.
- На `HomeScreen` есть индикатор успешного входа и кнопка `Выйти`.
- Логин пользователя отображается на экране для проверки потока авторизации.


## Кардинальный режим (сделано)

Если WebView/эмулятор ведут себя нестабильно, приложение открывает портал во внешнем браузере (`url_launcher`).

- Это самый быстрый способ получить рабочую авторизацию как на сайте.
- URL берётся из `API_BASE_URL` (по умолчанию `http://10.0.2.2:8000`).

Запуск:

```bash
flutter run --dart-define=API_BASE_URL=https://intech-forecast.com
```
