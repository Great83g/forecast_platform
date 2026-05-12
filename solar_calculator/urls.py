from django.urls import path

from . import views

app_name = "solar_calculator"

urlpatterns = [
    path("", views.calculator_page, name="page"),
    path("api/calculator/calculate/", views.calculate_api, name="calculate"),
    path("api/calculator/lead/", views.lead_api, name="lead"),
    path("solar-panels-kazakhstan/", views.seo_solar_panels_kazakhstan, name="seo-panels-kz"),
    path("solar-panels-almaty-price/", views.seo_solar_panels_almaty, name="seo-panels-almaty"),
    path("sell-electricity-kazakhstan/", views.seo_sell_electricity_kz, name="seo-sell-electricity"),
    path("solar-580w-panels/", views.seo_solar_580w, name="seo-580w"),
]
