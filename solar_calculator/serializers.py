from rest_framework import serializers

from .services.calculator_engine import CALC_MODES


class CalculatorRequestSerializer(serializers.Serializer):
    mode = serializers.ChoiceField(choices=CALC_MODES)
    inputs = serializers.DictField(required=True)
