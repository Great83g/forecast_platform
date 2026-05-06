from rest_framework import serializers

from .services.calculator_engine import CALC_MODES


class CalculatorRequestSerializer(serializers.Serializer):
    mode = serializers.ChoiceField(choices=CALC_MODES)
    inputs = serializers.DictField(required=True)


class LeadRequestSerializer(serializers.Serializer):
    name = serializers.CharField(max_length=120)
    phone = serializers.CharField(max_length=40)
    email = serializers.EmailField(required=False, allow_blank=True)
    comment = serializers.CharField(required=False, allow_blank=True)
    selected_plan = serializers.CharField(required=False, allow_blank=True)
    price = serializers.CharField(required=False, allow_blank=True)
    panel_count = serializers.CharField(required=False, allow_blank=True)
    system_power_kw = serializers.CharField(required=False, allow_blank=True)
    payback_years = serializers.CharField(required=False, allow_blank=True)
