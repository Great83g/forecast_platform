# stations/urls.py

from django.urls import path

from solar.views import UploadHistoryView

from .views import OrganizationListCreateView, OrganizationMemberListCreateView, StationListCreateView

urlpatterns = [
    # список / создание организаций
    path("orgs/", OrganizationListCreateView.as_view(), name="org-list-create"),
    path("orgs/<int:org_id>/members/", OrganizationMemberListCreateView.as_view(), name="org-members-list-create"),

    # список / создание станций
    path("stations/", StationListCreateView.as_view(), name="station-list-create"),

    # загрузка истории по станции
    path(
        "stations/<int:station_id>/upload_history/",
        UploadHistoryView.as_view(),
        name="upload_history",
    ),
]
