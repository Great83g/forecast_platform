from datetime import timedelta

from django.contrib.auth.models import User
from django.contrib.auth.password_validation import validate_password
from django.utils import timezone
from rest_framework import serializers

from .hcaptcha import verify_hcaptcha
from stations.models import Organization, OrganizationMember


class RegisterSerializer(serializers.ModelSerializer):
    password = serializers.CharField(write_only=True, required=True, validators=[validate_password])
    organization_name = serializers.CharField(write_only=True, required=False, allow_blank=True, max_length=200)
    hcaptcha_token = serializers.CharField(write_only=True, required=True)

    class Meta:
        model = User
        fields = ("username", "email", "password", "organization_name", "hcaptcha_token")

    def validate_hcaptcha_token(self, value):
        request = self.context.get("request")
        remote_ip = request.META.get("REMOTE_ADDR") if request else None
        ok, message = verify_hcaptcha(token=value, remote_ip=remote_ip)
        if not ok:
            raise serializers.ValidationError(message)
        return value

    def create(self, validated_data):
        validated_data.pop("hcaptcha_token", "")
        organization_name = validated_data.pop("organization_name", "")
        user = User.objects.create_user(
            username=validated_data["username"],
            email=validated_data.get("email"),
            password=validated_data["password"],
        )
        organization = Organization.objects.create(
            name=organization_name or f"{user.username} Workspace",
            owner=user,
            trial_ends_at=timezone.now() + timedelta(days=7),
        )
        OrganizationMember.objects.create(
            organization=organization,
            user=user,
            role=OrganizationMember.ROLE_OWNER,
        )
        return user


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = ("id", "username", "email")
