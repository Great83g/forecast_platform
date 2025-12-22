from django.contrib import admin
from django.urls import path, include
from django.views.generic import RedirectView
from django.contrib.auth import views as auth_views

urlpatterns = [
    path("admin/", admin.site.urls),

    # auth urls (password reset etc.)
    path("accounts/", include("django.contrib.auth.urls")),

    # главный логин/логаут под твой LOGIN_URL
    path("login/", auth_views.LoginView.as_view(template_name="accounts/login.html"), name="login"),
    path("logout/", auth_views.LogoutView.as_view(), name="logout"),

    # dashboard
    path("dashboard/", include(("dashboard.urls", "dashboard"), namespace="dashboard")),

    # корень -> дашборд
    path("", RedirectView.as_view(pattern_name="dashboard:station-list", permanent=False)),
]


