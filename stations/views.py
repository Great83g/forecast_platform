from datetime import timedelta

from django.shortcuts import get_object_or_404, render
from django.utils import timezone
from rest_framework import generics, permissions
from rest_framework.exceptions import PermissionDenied
from rest_framework.response import Response

from .models import Organization, OrganizationInvitation, OrganizationMember, Station
from .serializers import (
    InvitationAcceptSerializer,
    OrganizationInvitationSerializer,
    OrganizationMemberSerializer,
    OrganizationSerializer,
    StationSerializer,
)


def _get_actor_membership_or_403(org, user):
    membership = get_object_or_404(OrganizationMember, organization=org, user=user)
    if membership.role not in {OrganizationMember.ROLE_OWNER, OrganizationMember.ROLE_ADMIN}:
        raise PermissionDenied("Только owner/admin могут выполнять это действие.")
    return membership


class OrganizationListCreateView(generics.ListCreateAPIView):
    serializer_class = OrganizationSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return Organization.objects.filter(memberships__user=self.request.user).distinct()

    def perform_create(self, serializer):
        organization = serializer.save(
            owner=self.request.user,
            trial_ends_at=timezone.now() + timedelta(days=7),
        )
        OrganizationMember.objects.get_or_create(
            organization=organization,
            user=self.request.user,
            defaults={"role": OrganizationMember.ROLE_OWNER},
        )


class OrganizationMemberListCreateView(generics.ListCreateAPIView):
    serializer_class = OrganizationMemberSerializer
    permission_classes = [permissions.IsAuthenticated]

    def _organization(self):
        org_id = self.kwargs["org_id"]
        return get_object_or_404(Organization.objects.filter(memberships__user=self.request.user).distinct(), pk=org_id)

    def get_queryset(self):
        org = self._organization()
        return OrganizationMember.objects.filter(organization=org).select_related("user", "organization")

    def perform_create(self, serializer):
        org = self._organization()
        _get_actor_membership_or_403(org, self.request.user)
        serializer.save(organization=org)


class OrganizationInvitationListCreateView(generics.ListCreateAPIView):
    serializer_class = OrganizationInvitationSerializer
    permission_classes = [permissions.IsAuthenticated]

    def _organization(self):
        org_id = self.kwargs["org_id"]
        return get_object_or_404(Organization.objects.filter(memberships__user=self.request.user).distinct(), pk=org_id)

    def get_queryset(self):
        org = self._organization()
        return OrganizationInvitation.objects.filter(organization=org).select_related("invited_by", "accepted_by")

    def perform_create(self, serializer):
        org = self._organization()
        _get_actor_membership_or_403(org, self.request.user)

        invited_email = serializer.validated_data["invited_email"].lower()
        role = serializer.validated_data["role"]

        if OrganizationMember.objects.filter(organization=org, user__email__iexact=invited_email).exists():
            raise PermissionDenied("Пользователь с таким email уже состоит в организации.")

        OrganizationInvitation.objects.filter(
            organization=org,
            invited_email__iexact=invited_email,
            status=OrganizationInvitation.STATUS_PENDING,
        ).update(status=OrganizationInvitation.STATUS_CANCELLED)

        serializer.save(
            organization=org,
            invited_email=invited_email,
            role=role,
            invited_by=self.request.user,
            expires_at=timezone.now() + timedelta(days=7),
        )


class OrganizationInvitationAcceptView(generics.GenericAPIView):
    serializer_class = InvitationAcceptSerializer
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        invitation = serializer.save()
        return Response(
            {
                "status": "ok",
                "organization": invitation.organization_id,
                "role": invitation.role,
                "accepted_at": invitation.accepted_at,
            }
        )


class StationListCreateView(generics.ListCreateAPIView):
    serializer_class = StationSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return Station.objects.filter(org__memberships__user=self.request.user).distinct()

    def perform_create(self, serializer):
        org = serializer.validated_data["org"]
        if not OrganizationMember.objects.filter(organization=org, user=self.request.user).exists():
            raise PermissionDenied("Вы не состоите в этой организации.")
        serializer.save()

# Create your views here.
