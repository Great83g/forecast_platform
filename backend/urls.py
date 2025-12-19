# backend/urls.py

from django.contrib import admin
from django.urls import path, include
from django.views.generic import RedirectView
from django.contrib.auth import views as auth_views

urlpatterns = [
    # Django admin
    path("admin/", admin.site.urls),

    # API (JWT и т.п.)
    path("api/auth/", include("accounts.urls")),   # логин/регистрация по API
    path("api/", include("stations.urls")),        # станции, orgs и т.д.

    # Веб-дашборд
    path("dashboard/", include("dashboard.urls")),

    # Корень сайта → список станций дашборда
    path(
        "",
        RedirectView.as_view(
            pattern_name="dashboard-station-list",
            permanent=False,
        ),
        name="root",
    ),
] + [
    # Страница логина
    path(
        "login/",
        auth_views.LoginView.as_view(
            template_name="accounts/login.html",
        ),
        name="login",
    ),
    # Выход из системы
    path(
        "logout/",
        auth_views.LogoutView.as_view(),
        name="logout",
    ),
]
