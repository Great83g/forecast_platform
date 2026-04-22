from django.contrib.auth.decorators import login_required
from django.shortcuts import render
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response

from .serializers import CalculatorRequestSerializer
from .services.calculator_engine import calculate


@login_required
def calculator_page(request):
    return render(request, "solar_calculator/calculator_page.html")


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def calculate_api(request):
    serializer = CalculatorRequestSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)
    payload = serializer.validated_data
    try:
        output = calculate(payload["mode"], payload.get("inputs", {}))
    except ValueError as exc:
        return Response({"detail": str(exc)}, status=status.HTTP_400_BAD_REQUEST)
    return Response(output)
