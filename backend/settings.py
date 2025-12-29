"""
Django settings for backend project.
"""

from pathlib import Path

# === БАЗОВЫЙ ПУТЬ ===
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models_cache"


# === Visual Crossing API ===
VISUAL_CROSSING_API_KEY = "WFZVPPR44XXZALVNSDDWDALPU"

# === Email (SMTP) ===
EMAIL_BACKEND = "django.core.mail.backends.smtp.EmailBackend"
EMAIL_HOST = "mail.care-tech.kz"
EMAIL_PORT = 587
EMAIL_USE_TLS = True
EMAIL_HOST_USER = "zhezsolar@tgs-energy.kz"
EMAIL_HOST_PASSWORD = "Great@creat"
DEFAULT_FROM_EMAIL = EMAIL_HOST_USER

# === Open-Meteo API ===
OPEN_METEO_BASE_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_TIMEOUT = 45

# === Forecast weather providers (priority order) ===
FORECAST_WEATHER_PROVIDERS = [
    "visual_crossing",
    "open_meteo",
]


# === БЕЗОПАСНОСТЬ (для разработки) ===
SECRET_KEY = "django-insecure-%v0pjz8ji+z50*r6xlldh55l%n2u@_7r%4j0pvxm26e%5o2rr*"
DEBUG = True

ALLOWED_HOSTS = [
    "127.0.0.1",
    "localhost",
]

# === ПРИЛОЖЕНИЯ ===
INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",

    "rest_framework",

    "accounts",
    "stations",
    "solar",
    "dashboard",
]

# === MIDDLEWARE ===
MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "backend.urls"

# === ШАБЛОНЫ ===
TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "backend.wsgi.application"

# === БАЗА ДАННЫХ ===
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

# === ВАЛИДАЦИЯ ПАРОЛЕЙ ===
AUTH_PASSWORD_VALIDATORS = [
    {
        "NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.MinimumLengthValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.CommonPasswordValidator",
    },
    {
        "NAME": "django.contrib.auth.password_validation.NumericPasswordValidator",
    },
]

# === ЛОКАЛИЗАЦИЯ ===
LANGUAGE_CODE = "ru-ru"
TIME_ZONE = "Asia/Almaty"

USE_I18N = True
USE_TZ = True

# === СТАТИКА ===
STATIC_URL = "/static/"

MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# === DRF / JWT ===
REST_FRAMEWORK = {
    "DEFAULT_AUTHENTICATION_CLASSES": (
        "rest_framework.authentication.SessionAuthentication",
        "rest_framework.authentication.BasicAuthentication",
        "rest_framework_simplejwt.authentication.JWTAuthentication",
    ),
}

# === АУТЕНТИФИКАЦИЯ ===
LOGIN_URL = "/login/"
LOGIN_REDIRECT_URL = "/dashboard/"
LOGOUT_REDIRECT_URL = "/login/"
