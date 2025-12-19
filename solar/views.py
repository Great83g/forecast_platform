# solar/views.py

from __future__ import annotations

import pandas as pd
from django.shortcuts import get_object_or_404
from django.db import transaction

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
    API-загрузка истории для станции.

    URL (см. stations/urls.py) что-то вроде:
        /api/stations/<pk>/upload-history/

    Ожидает multipart/form-data с полем "file":
      - CSV или XLSX
      - обязательные колонки:
            ds          – дата/время
            Irradiation – радиация
            Air_Temp    – температура воздуха
            PV_Temp     – температура панелей
            Power_kW    – фактическая выработка (кВт)

    Возвращает JSON:
        {
          "status": "ok",
          "station": <id>,
          "imported_rows": <сколько строк записано/обновлено>
        }
    или:
        { "status": "error", "message": "..." }
    """

    parser_classes = (MultiPartParser, FormParser)
    permission_classes = (IsAuthenticated,)
    serializer_class = HistoryUploadSerializer

    def post(self, request, pk: int, *args, **kwargs):
        station = get_object_or_404(Station, pk=pk)

        # сериализатор просто валидирует наличие файла
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        upload_file = serializer.validated_data["file"]

        filename = upload_file.name.lower()

        # ---------- читаем файл в pandas ----------
        try:
            if filename.endswith(".csv"):
                df = pd.read_csv(upload_file)
            elif filename.endswith((".xlsx", ".xls")):
                df = pd.read_excel(upload_file)
            else:
                return Response(
                    {
                        "status": "error",
                        "message": "Поддерживаются только файлы .csv или .xlsx",
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )
        except Exception as e:
            return Response(
                {
                    "status": "error",
                    "message": f"Ошибка чтения файла: {e}",
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        # ---------- проверяем колонки ----------
        required_cols = ["ds", "Irradiation", "Air_Temp", "PV_Temp", "Power_kW"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return Response(
                {
                    "status": "error",
                    "message": f"Отсутствуют обязательные колонки: {missing}",
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        # ---------- парсим даты ----------
        try:
            df["ds"] = pd.to_datetime(df["ds"])
        except Exception as e:
            return Response(
                {
                    "status": "error",
                    "message": f"Не удалось распарсить колонку 'ds' как дату: {e}",
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        df = df.replace({pd.NA: None})

        created = 0

        # ---------- пишем в базу ----------
        with transaction.atomic():
            for _, row in df.iterrows():
                ts = row["ds"]

                # update_or_create по (station, timestamp)
                SolarRecord.objects.update_or_create(
                    station=station,
                    timestamp=ts,
                    defaults={
                        "irradiation": row["Irradiation"],
                        "air_temp": row["Air_Temp"],
                        "pv_temp": row["PV_Temp"],
                        "power_kw": row["Power_KW"],
                    },
                )
                created += 1

        return Response(
            {
                "status": "ok",
                "station": station.id,
                "imported_rows": int(created),
            },
            status=status.HTTP_201_CREATED,
        )
