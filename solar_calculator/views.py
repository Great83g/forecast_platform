from django.shortcuts import render
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response

from .serializers import CalculatorRequestSerializer
from .services.calculator_engine import calculate


def calculator_page(request):
    return render(request, "solar_calculator/calculator_page.html")


@api_view(["POST"])
@permission_classes([AllowAny])
def calculate_api(request):
    serializer = CalculatorRequestSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)
    payload = serializer.validated_data
    output = calculate(payload["mode"], payload["inputs"])
    return Response(output)
