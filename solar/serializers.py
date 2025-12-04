# solar/serializers.py

from rest_framework import serializers


class HistoryUploadSerializer(serializers.Serializer):
    file = serializers.FileField()

