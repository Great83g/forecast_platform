from django.contrib import admin
from django.conf import settings
from django.conf.urls.static import static
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from django.urls import path, include
from django.views.generic import RedirectView
from django.contrib.auth import views as auth_views
from accounts.web_views import RegisterPageView, LoginPageView
from backend.branding_views import brand_logo

urlpatterns = [
    path("admin/", admin.site.urls),

    # auth urls (password reset etc.)
    path("accounts/", include("django.contrib.auth.urls")),

    # главный логин/логаут под твой LOGIN_URL
    path("login/", LoginPageView.as_view(), name="login"),
    path("logout/", auth_views.LogoutView.as_view(), name="logout"),
    path("register/", RegisterPageView.as_view(), name="register"),

    path("brand-logo/", brand_logo, name="brand-logo"),

    # dashboard
    path("dashboard/", include(("dashboard.urls", "dashboard"), namespace="dashboard")),

    # wind forecast module
    path("wind/", include(("wind.urls", "wind"), namespace="wind")),

    # API
    path("api/accounts/", include("accounts.urls")),
    path("api/", include("stations.urls")),

    # корень -> дашборд
    path("", RedirectView.as_view(pattern_name="dashboard:station-list", permanent=False)),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

# Serve static assets (including Django admin) via Django when no external static server is configured.
urlpatterns += staticfiles_urlpatterns()
