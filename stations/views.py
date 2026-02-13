from datetime import timedelta

from django.conf import settings
from django.core.mail import send_mail
from django.shortcuts import get_object_or_404, render
from django.utils import timezone
from rest_framework import generics, permissions, status
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


INVITATION_THROTTLE_LIMIT_PER_HOUR = 10


def _get_actor_membership_or_403(org, user):
    membership = get_object_or_404(OrganizationMember, organization=org, user=user)
    if membership.role not in {OrganizationMember.ROLE_OWNER, OrganizationMember.ROLE_ADMIN}:
        raise PermissionDenied("Только owner/admin могут выполнять это действие.")
    return membership



def _ensure_org_write_access_or_403(org):
    if org.can_write():
        return
    raise PermissionDenied(org.write_access_reason())


def _send_invitation_email(invitation):
    base_url = getattr(settings, "PORTAL_BASE_URL", "")
    accept_url = f"{base_url.rstrip('/')}/org/invitations/accept/{invitation.token}" if base_url else invitation.token
    body = (
        f"Вас пригласили в организацию '{invitation.organization.name}' с ролью '{invitation.role}'.\n"
        f"Ссылка для принятия приглашения: {accept_url}\n"
        f"Срок действия: {invitation.expires_at:%Y-%m-%d %H:%M}."
    )
    send_mail(
        subject=f"Приглашение в организацию {invitation.organization.name}",
        message=body,
        from_email=getattr(settings, "DEFAULT_FROM_EMAIL", None),
        recipient_list=[invitation.invited_email],
        fail_silently=True,
    )


class OrganizationListCreateView(generics.ListCreateAPIView):
    serializer_class = OrganizationSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return Organization.objects.filter(memberships__user=self.request.user).distinct()

    def perform_create(self, serializer):
        organization = serializer.save(
            owner=self.request.user,
            trial_ends_at=timezone.now() + timedelta(days=7),
            subscription_status=Organization.SUBSCRIPTION_TRIALING,
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
        _ensure_org_write_access_or_403(org)
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
        _ensure_org_write_access_or_403(org)

        window_start = timezone.now() - timedelta(hours=1)
        recent_invites_count = OrganizationInvitation.objects.filter(
            organization=org,
            invited_by=self.request.user,
            created_at__gte=window_start,
        ).count()
        if recent_invites_count >= INVITATION_THROTTLE_LIMIT_PER_HOUR:
            raise PermissionDenied("Слишком много приглашений за последний час. Попробуйте позже.")

        invited_email = serializer.validated_data["invited_email"].lower()
        role = serializer.validated_data["role"]

        if OrganizationMember.objects.filter(organization=org, user__email__iexact=invited_email).exists():
            raise PermissionDenied("Пользователь с таким email уже состоит в организации.")

        OrganizationInvitation.objects.filter(
            organization=org,
            invited_email__iexact=invited_email,
            status=OrganizationInvitation.STATUS_PENDING,
        ).update(status=OrganizationInvitation.STATUS_CANCELLED)

        invitation = serializer.save(
            organization=org,
            invited_email=invited_email,
            role=role,
            invited_by=self.request.user,
            expires_at=timezone.now() + timedelta(days=7),
        )
        _send_invitation_email(invitation)


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


class OrganizationInvitationResendView(generics.GenericAPIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, org_id, invitation_id, *args, **kwargs):
        org = get_object_or_404(Organization.objects.filter(memberships__user=request.user).distinct(), pk=org_id)
        _get_actor_membership_or_403(org, request.user)
        _ensure_org_write_access_or_403(org)

        invitation = get_object_or_404(OrganizationInvitation, pk=invitation_id, organization=org)
        if invitation.status != OrganizationInvitation.STATUS_PENDING:
            raise PermissionDenied("Можно переотправить только pending-приглашение.")

        invitation.expires_at = timezone.now() + timedelta(days=7)
        invitation.save(update_fields=["expires_at"])
        _send_invitation_email(invitation)
        return Response({"status": "resent", "invitation_id": invitation.id})


class OrganizationInvitationRevokeView(generics.GenericAPIView):
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request, org_id, invitation_id, *args, **kwargs):
        org = get_object_or_404(Organization.objects.filter(memberships__user=request.user).distinct(), pk=org_id)
        _get_actor_membership_or_403(org, request.user)
        _ensure_org_write_access_or_403(org)

        invitation = get_object_or_404(OrganizationInvitation, pk=invitation_id, organization=org)
        if invitation.status != OrganizationInvitation.STATUS_PENDING:
            return Response({"status": invitation.status}, status=status.HTTP_200_OK)

        invitation.status = OrganizationInvitation.STATUS_CANCELLED
        invitation.save(update_fields=["status"])
        return Response({"status": "cancelled", "invitation_id": invitation.id})


class StationListCreateView(generics.ListCreateAPIView):
    serializer_class = StationSerializer
    permission_classes = [permissions.IsAuthenticated]

    def get_queryset(self):
        return Station.objects.filter(org__memberships__user=self.request.user).distinct()

    def perform_create(self, serializer):
        org = serializer.validated_data["org"]
        if not OrganizationMember.objects.filter(organization=org, user=self.request.user).exists():
            raise PermissionDenied("Вы не состоите в этой организации.")
        _ensure_org_write_access_or_403(org)
        serializer.save()

# Create your views here.
