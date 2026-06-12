from django.urls import path

from . import views

app_name = "virtual_ess"

urlpatterns = [
    path("", views.station_list, name="station-list"),
    path("station/<int:station_id>/settings/", views.station_settings, name="station-settings"),
    path("station/<int:station_id>/simulate/", views.station_simulate, name="station-simulate"),
    path("run/<int:run_id>/", views.run_detail, name="run-detail"),
]
