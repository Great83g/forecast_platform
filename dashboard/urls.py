# dashboard/urls.py

from django.urls import path
from .views import (
    station_list,
    station_detail,
    station_create,
    station_upload_history,
    station_export_history,
    station_train_models,
    run_station_forecast,     # запуск прогноза
    station_forecast_list,    # просмотр прогноза
    station_export_forecast,  # экспорт прогноза
)

urlpatterns = [

    # --- СПИСОК СТАНЦИЙ ---
    path("", station_list, name="dashboard-station-list"),

    # --- СОЗДАНИЕ СТАНЦИИ ---
    path("station/add/", station_create, name="dashboard-station-create"),

    # --- ДЕТАЛЬНАЯ СТРАНИЦА ---
    path("station/<int:pk>/", station_detail, name="dashboard-station-detail"),

    # --- ИСТОРИЯ ---
    path("station/<int:pk>/upload/", station_upload_history, name="dashboard-station-upload"),
    path("station/<int:pk>/export/", station_export_history, name="dashboard-station-export"),

    # --- ОБУЧЕНИЕ МОДЕЛЕЙ ---
    path("station/<int:pk>/train/", station_train_models, name="dashboard-station-train"),

    # --- ПРОГНОЗ ---
    path("station/<int:pk>/forecast/", run_station_forecast, name="dashboard-station-forecast"),

    # --- НОВОЕ: ПРОСМОТР ПРОГНОЗА ---
    path(
        "station/<int:pk>/forecast/list/",
        station_forecast_list,
        name="dashboard-station-forecast-list"
    ),

    # --- НОВОЕ: ЭКСПОРТ ПРОГНОЗА ---
    path(
        "station/<int:pk>/forecast/export/",
        station_export_forecast,
        name="dashboard-station-export-forecast"
    ),
]

