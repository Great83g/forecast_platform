from django.conf import settings


def project_settings(request):
    return {
        "LOGIN_HERO_VIDEO_URL": getattr(settings, "LOGIN_HERO_VIDEO_URL", "/media/login-hero.mp4"),
    }
