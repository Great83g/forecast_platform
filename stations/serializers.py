from django.utils import timezone
from rest_framework import serializers

from .models import Organization, OrganizationInvitation, OrganizationMember, Station


class OrganizationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Organization
        fields = ["id", "name", "created_at", "trial_ends_at", "is_active", "subscription_status"]
        read_only_fields = ["created_at", "trial_ends_at", "is_active", "subscription_status"]


class OrganizationMemberSerializer(serializers.ModelSerializer):
    username = serializers.CharField(source="user.username", read_only=True)
    email = serializers.CharField(source="user.email", read_only=True)

    class Meta:
        model = OrganizationMember
        fields = ["id", "organization", "user", "username", "email", "role", "created_at"]
        read_only_fields = ["created_at", "username", "email"]


class OrganizationInvitationSerializer(serializers.ModelSerializer):
    invited_by_username = serializers.CharField(source="invited_by.username", read_only=True)

    class Meta:
        model = OrganizationInvitation
        fields = [
            "id",
            "organization",
            "invited_email",
            "role",
            "status",
            "token",
            "invited_by",
            "invited_by_username",
            "accepted_by",
            "expires_at",
            "accepted_at",
            "created_at",
        ]
        read_only_fields = ["status", "token", "invited_by", "accepted_by", "accepted_at", "created_at"]


class InvitationAcceptSerializer(serializers.Serializer):
    token = serializers.CharField(max_length=64)

    def validate_token(self, value):
        try:
            invitation = OrganizationInvitation.objects.select_related("organization").get(token=value)
        except OrganizationInvitation.DoesNotExist as exc:
            raise serializers.ValidationError("Приглашение не найдено.") from exc

        if invitation.status != OrganizationInvitation.STATUS_PENDING:
            raise serializers.ValidationError("Приглашение уже неактивно.")

        if invitation.is_expired:
            raise serializers.ValidationError("Срок действия приглашения истёк.")

        return value

    def save(self, **kwargs):
        user = self.context["request"].user
        invitation = OrganizationInvitation.objects.select_related("organization").get(token=self.validated_data["token"])

        if not user.email or user.email.lower() != invitation.invited_email.lower():
            raise serializers.ValidationError({"token": "Email аккаунта не совпадает с email приглашения."})

        OrganizationMember.objects.update_or_create(
            organization=invitation.organization,
            user=user,
            defaults={"role": invitation.role},
        )

        invitation.status = OrganizationInvitation.STATUS_ACCEPTED
        invitation.accepted_by = user
        invitation.accepted_at = invitation.accepted_at or timezone.now()
        invitation.save(update_fields=["status", "accepted_by", "accepted_at"])

        return invitation


class StationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Station
        fields = [
            "id",
            "name",
            "capacity_mw",
            "latitude",
            "longitude",
            "timezone",
            "data_shift_hours",
            "org",
            "history_source",
            "history_scale_by_capacity",
            "auto_history_enabled",
            "auto_history_folder",
            "auto_history_script",
            "auto_history_run_time",
            "auto_history_last_run_date",
        ]
        read_only_fields = ["auto_history_last_run_date"]
        extra_kwargs = {
            "data_shift_hours": {"required": False},
        }

    def validate(self, attrs):
        org = attrs.get("org") or getattr(self.instance, "org", None)
        request = self.context.get("request")
        if org is None or request is None:
            return attrs

        if not OrganizationMember.objects.filter(organization=org, user=request.user).exists():
            raise serializers.ValidationError({"org": "Вы не состоите в этой организации."})

        history_source = attrs.get("history_source")
        if history_source and history_source.org_id != org.id:
            raise serializers.ValidationError({"history_source": "Источник истории должен быть из той же организации."})

        return attrs
