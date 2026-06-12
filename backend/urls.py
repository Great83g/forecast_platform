from django.contrib import admin
from django.conf import settings
from django.conf.urls.static import static
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from django.urls import path, include
from django.views.generic import RedirectView
from django.contrib.auth import views as auth_views
from accounts.web_views import RegisterPageView, LoginPageView
from backend.branding_views import brand_logo
from solar_calculator import views as solar_views

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

    # Virtual ESS / EMS Balance Simulator
    path("virtual-ess/", include(("virtual_ess.urls", "virtual_ess"), namespace="virtual_ess")),

    # API
    path("api/accounts/", include("accounts.urls")),
    path("api/assistant/", include(("ai_assistant.urls", "ai_assistant"), namespace="ai_assistant")),
    path("api/", include("stations.urls")),

    # solar calculator module
    path("solar-calculator/", include(("solar_calculator.urls", "solar_calculator"), namespace="solar_calculator")),
    path("solar-calculator", solar_views.calculator_page),
    path("solar-panels-kazakhstan", solar_views.seo_solar_panels_kazakhstan),
    path("solar-panels-almaty-price", solar_views.seo_solar_panels_almaty),
    path("sell-electricity-kazakhstan", solar_views.seo_sell_electricity_kz),
    path("solar-580w-panels", solar_views.seo_solar_580w),
    path("sitemap.xml", solar_views.sitemap_xml),
    path("robots.txt", solar_views.robots_txt),

    # корень -> дашборд
    path("", RedirectView.as_view(pattern_name="dashboard:station-list", permanent=False)),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

# Serve static assets (including Django admin) via Django when no external static server is configured.
urlpatterns += staticfiles_urlpatterns()
