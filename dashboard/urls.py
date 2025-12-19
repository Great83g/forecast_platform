from django.urls import path
from . import views

app_name = "dashboard"

urlpatterns = [
    path("", views.station_list, name="station-list"),

    path("station/create/", views.station_create, name="station-create"),
    path("station/<int:pk>/edit/", views.station_edit, name="station-edit"),
    path("station/<int:pk>/", views.station_detail, name="station-detail"),

    # История + загрузка + экспорт
    path("station/<int:pk>/upload/", views.station_upload, name="station-upload"),
    path("station/<int:pk>/export-history/", views.station_export_history, name="station-export-history"),

    # Обучение и прогноз
    path("station/<int:pk>/train/", views.station_train, name="station-train"),
    path("station/<int:pk>/train-models/", views.station_train, name="station-train-models"),

    path("station/<int:pk>/forecast/list/", views.station_forecast_list, name="station-forecast-list"),
    path("station/<int:pk>/forecast/run/", views.station_forecast_run, name="station-forecast-run"),
    path("station/<int:pk>/forecast/export/", views.station_forecast_export, name="station-forecast-export"),
    path("station/<int:pk>/forecast/clear/", views.station_forecast_clear, name="station-forecast-clear"),
]

