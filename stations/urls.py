# stations/urls.py

from django.urls import path

from solar.views import UploadHistoryView

from .views import (
    OrganizationInvitationAcceptView,
    OrganizationInvitationListCreateView,
    OrganizationInvitationResendView,
    OrganizationInvitationRevokeView,
    OrganizationListCreateView,
    OrganizationMemberListCreateView,
    StationListCreateView,
)

urlpatterns = [
    # список / создание организаций
    path("orgs/", OrganizationListCreateView.as_view(), name="org-list-create"),
    path("orgs/<int:org_id>/members/", OrganizationMemberListCreateView.as_view(), name="org-members-list-create"),
    path("orgs/<int:org_id>/invitations/", OrganizationInvitationListCreateView.as_view(), name="org-invitations-list-create"),
    path("orgs/invitations/accept/", OrganizationInvitationAcceptView.as_view(), name="org-invitations-accept"),
    path("orgs/<int:org_id>/invitations/<int:invitation_id>/resend/", OrganizationInvitationResendView.as_view(), name="org-invitations-resend"),
    path("orgs/<int:org_id>/invitations/<int:invitation_id>/revoke/", OrganizationInvitationRevokeView.as_view(), name="org-invitations-revoke"),

    # список / создание станций
    path("stations/", StationListCreateView.as_view(), name="station-list-create"),

    # загрузка истории по станции
    path(
        "stations/<int:station_id>/upload_history/",
        UploadHistoryView.as_view(),
        name="upload_history",
    ),
]
