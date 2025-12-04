# stations/urls.py

from django.urls import path
from .views import (
    OrganizationListCreateView,
    StationListCreateView,
)
from solar.views import UploadHistoryView  # класс из solar.views

urlpatterns = [
    # список / создание организаций
    path("orgs/", OrganizationListCreateView.as_view(), name="org-list-create"),

    # список / создание станций
    path("stations/", StationListCreateView.as_view(), name="station-list-create"),

    # загрузка истории по станции
    path(
        "stations/<int:station_id>/upload_history/",
        UploadHistoryView.as_view(),
        name="upload_history",
    ),
]

