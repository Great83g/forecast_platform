from django.apps import AppConfig


class SolarConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'solar'

    def ready(self):
        from . import signals  # noqa: F401
