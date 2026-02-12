from rest_framework import serializers

from .models import Organization, OrganizationMember, Station


class OrganizationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Organization
        fields = ["id", "name", "created_at", "trial_ends_at", "is_active"]
        read_only_fields = ["created_at", "trial_ends_at", "is_active"]


class OrganizationMemberSerializer(serializers.ModelSerializer):
    username = serializers.CharField(source="user.username", read_only=True)
    email = serializers.CharField(source="user.email", read_only=True)

    class Meta:
        model = OrganizationMember
        fields = ["id", "organization", "user", "username", "email", "role", "created_at"]
        read_only_fields = ["created_at", "username", "email"]


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
            "org",
            "history_source",
            "history_scale_by_capacity",
        ]
