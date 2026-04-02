from django.urls import path

from . import views

app_name = "wind"

urlpatterns = [
    path("", views.station_list, name="station-list"),
    path("station/create/", views.station_create, name="station-create"),
    path("station/<int:pk>/", views.station_detail, name="station-detail"),
    path("station/<int:pk>/upload/", views.station_upload, name="station-upload"),
    path("station/<int:pk>/forecast/list/", views.station_forecast_list, name="station-forecast-list"),
    path("station/<int:pk>/train/", views.station_train, name="station-train"),
]
