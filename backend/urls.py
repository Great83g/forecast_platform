from django.contrib import admin
from django.urls import path, include
from django.views.generic import RedirectView


urlpatterns = [
    path("admin/", admin.site.urls),

    # auth (login/logout/password_reset...)
    path("accounts/", include("django.contrib.auth.urls")),

    # dashboard app (с namespace="dashboard")
    path("dashboard/", include(("dashboard.urls", "dashboard"), namespace="dashboard")),

    # корень сайта -> /dashboard/
    path("", RedirectView.as_view(pattern_name="dashboard:station-list", permanent=False)),
]

