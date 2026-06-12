"""
Django settings for backend project.
"""

from pathlib import Path
import os

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

_default_allowed_hosts = [
    "127.0.0.1",
    "localhost",
    "0.0.0.0",
    "intech-forecast.com",
    "www.intech-forecast.com",
]

_env_allowed_hosts = [h.strip() for h in os.getenv("ALLOWED_HOSTS", "").split(",") if h.strip()]
ALLOWED_HOSTS = _env_allowed_hosts or _default_allowed_hosts

_default_csrf_trusted_origins = [
    "https://intech-forecast.com",
    "https://www.intech-forecast.com",
    "http://127.0.0.1",
    "http://localhost",
]
_env_csrf_trusted_origins = [
    origin.strip() for origin in os.getenv("CSRF_TRUSTED_ORIGINS", "").split(",") if origin.strip()
]
CSRF_TRUSTED_ORIGINS = _env_csrf_trusted_origins or _default_csrf_trusted_origins

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
    "wind",
    "solar_calculator",
    "ai_assistant",
    "virtual_ess",
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
                "backend.context_processors.project_settings",
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
        "OPTIONS": {
            # Смягчает intermittent `database is locked` при конкурентных write-запросах
            # (web + shell/management commands).
            "timeout": int(os.getenv("SQLITE_TIMEOUT_SECONDS", "30")),
        },
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

STATIC_ROOT = BASE_DIR / "staticfiles"

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

# === Login page hero media ===
# Can be overridden on server without template edits:
# LOGIN_HERO_VIDEO_URL=/media/login-hero.mp4
# LOGIN_HERO_POSTER_URL=/media/login-background.png
LOGIN_HERO_VIDEO_URL = os.getenv("LOGIN_HERO_VIDEO_URL", "/media/login-hero.mp4")
LOGIN_HERO_POSTER_URL = os.getenv("LOGIN_HERO_POSTER_URL", "/media/login-background.png")

# === hCaptcha ===
HCAPTCHA_SITE_KEY = os.getenv("HCAPTCHA_SITE_KEY", "")
HCAPTCHA_SECRET_KEY = os.getenv("HCAPTCHA_SECRET_KEY", "")

# === Bitrix24 ===
BITRIX_WEBHOOK_URL = os.getenv("BITRIX_WEBHOOK_URL", "https://portal.care-tech.kz/rest/1/alg2q93ky7mw2oly/")


# === AI Assistant LLM fallback ===
AI_ASSISTANT_LLM_ENABLED = os.getenv("AI_ASSISTANT_LLM_ENABLED", "false").lower() == "true"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_TIMEOUT_SECONDS = int(os.getenv("OPENAI_TIMEOUT_SECONDS", "20"))
