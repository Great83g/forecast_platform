from rest_framework import serializers
from .models import Organization, Station


class OrganizationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Organization
        fields = ["id", "name"]


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
