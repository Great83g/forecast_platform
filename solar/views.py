# solar/views.py

import pandas as pd

from rest_framework.generics import GenericAPIView
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status

from stations.models import Station
from .models import SolarRecord
from .serializers import HistoryUploadSerializer


class UploadHistoryView(GenericAPIView):
    """
    Загрузка истории по станции из CSV/Excel.

    Ожидаемый формат колонок:
    - ds          : дата-время (строка, парсится в pandas.to_datetime)
    - Irradiation : радиация (Вт/м² или кВт*ч/м², как у тебя в файле)
    - Air_Temp    : температура воздуха
    - PV_Temp     : температура панелей (можно пусто)
    - Power_kW    : фактическая мощность/выработка в кВт
    """

    permission_classes = [IsAuthenticated]
    parser_classes = (MultiPartParser, FormParser)
    serializer_class = HistoryUploadSerializer

    def post(self, request, station_id, *args, **kwargs):
        # DRF сам возьмёт HistoryUploadSerializer через get_serializer()
        serializer = self.get_serializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        # Ищем станцию
        try:
            station = Station.objects.get(id=station_id)
        except Station.DoesNotExist:
            return Response(
                {"detail": "Station not found"},
                status=status.HTTP_404_NOT_FOUND,
            )

        file = serializer.validated_data["file"]
        filename = file.name.lower()

        # Читаем CSV/Excel
        try:
            if filename.endswith(".csv"):
                df = pd.read_csv(file)
            elif filename.endswith(".xlsx") or filename.endswith(".xls"):
                df = pd.read_excel(file)
            else:
                return Response(
                    {"detail": "Unsupported file type. Use .csv or .xlsx"},
                    status=status.HTTP_400_BAD_REQUEST,
                )
        except Exception as e:
            return Response(
                {"detail": f"Error reading file: {e}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        required_cols = ["ds", "Irradiation", "Air_Temp", "PV_Temp", "Power_kW"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return Response(
                {"detail": f"Missing columns: {missing}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # Парсим даты
        try:
            df["ds"] = pd.to_datetime(df["ds"])
        except Exception as e:
            return Response(
                {"detail": f"Cannot parse 'ds' to datetime: {e}"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        created = 0
        for _, row in df.iterrows():
            SolarRecord.objects.update_or_create(
                station=station,
                timestamp=row["ds"],
                defaults={
                    "irradiation": row["Irradiation"],
                    "air_temp": row["Air_Temp"],
                    "pv_temp": row["PV_Temp"],
                    "power_kw": row["Power_kW"],
                },
            )
            created += 1

        return Response(
            {
                "status": "ok",
                "station": station.id,
                "imported_rows": created,
            },
            status=status.HTTP_201_CREATED,
        )

