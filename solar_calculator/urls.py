from django.urls import path

from . import views

app_name = "solar_calculator"

urlpatterns = [
    path("", views.calculator_page, name="page"),
    path("api/calculator/calculate/", views.calculate_api, name="calculate"),
]
