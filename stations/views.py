from datetime import timedelta

from django.shortcuts import get_object_or_404, render
from django.utils import timezone
from rest_framework import generics, permissions
from rest_framework.exceptions import PermissionDenied

from .models import Organization, OrganizationMember, Station
from .serializers import OrganizationMemberSerializer, OrganizationSerializer, StationSerializer


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
        actor_membership = get_object_or_404(OrganizationMember, organization=org, user=self.request.user)
        if actor_membership.role not in {OrganizationMember.ROLE_OWNER, OrganizationMember.ROLE_ADMIN}:
            raise PermissionDenied("Только owner/admin могут управлять участниками.")
        serializer.save(organization=org)


class StationListCreateView(generics.ListCreateAPIView):
    serializer_class = StationSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return Station.objects.filter(org__memberships__user=self.request.user).distinct()

# Create your views here.
