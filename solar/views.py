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
            Power_kW    – фактическая выработка (кВт)
      - радиация: Irradiation_GHI/GHI и/или Irradiation_POA/POA, либо старая Irradiation

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

    def post(self, request, station_id: int, *args, **kwargs):
        station = get_object_or_404(Station.objects.filter(org__memberships__user=request.user).distinct(), pk=station_id)

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

        # ---------- проверяем и маппим колонки ----------
        col_map = {str(c).strip().lower(): c for c in df.columns}

        def pick(*names):
            for name in names:
                key = name.strip().lower()
                if key in col_map:
                    return col_map[key]
            return None

        col_ds = pick("ds", "timestamp", "datetime", "date_time")
        col_power = pick("Power_kW", "Power_KW", "power_kw", "power", "y")
        col_legacy_irr = pick("Irradiation", "irradiation")
        col_ghi = pick("Irradiation_GHI", "irradiation_ghi", "GHI", "ghi")
        col_poa = pick("Irradiation_POA", "irradiation_poa", "POA", "poa")
        col_air = pick("Air_Temp", "air_temp", "air temperature", "temperature")
        col_pv = pick("PV_Temp", "pv_temp", "module_temp", "panel_temp")

        if not col_ds or not col_power:
            return Response(
                {
                    "status": "error",
                    "message": "Нужны колонки ds/timestamp и Power_kW/power_kw.",
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        # ---------- парсим даты ----------
        try:
            df[col_ds] = pd.to_datetime(df[col_ds])
        except Exception as e:
            return Response(
                {
                    "status": "error",
                    "message": f"Не удалось распарсить колонку даты как дату: {e}",
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        for col in [col_power, col_legacy_irr, col_ghi, col_poa, col_air, col_pv]:
            if col:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.replace({pd.NA: None})

        created = 0

        # ---------- пишем в базу ----------
        with transaction.atomic():
            for _, row in df.iterrows():
                ts = row[col_ds]
                legacy_irr = row[col_legacy_irr] if col_legacy_irr else None
                ghi = row[col_ghi] if col_ghi else None
                poa = row[col_poa] if col_poa else None
                if pd.isna(ghi) and pd.notna(legacy_irr) and station.irradiation_type == Station.IRRADIATION_GHI:
                    ghi = legacy_irr
                if pd.isna(poa) and pd.notna(legacy_irr) and station.irradiation_type == Station.IRRADIATION_POA:
                    poa = legacy_irr

                # update_or_create по (station, timestamp)
                SolarRecord.objects.update_or_create(
                    station=station,
                    timestamp=ts,
                    defaults={
                        "irradiation": legacy_irr if pd.notna(legacy_irr) else (ghi if pd.notna(ghi) else None),
                        "irradiation_ghi": ghi if pd.notna(ghi) else None,
                        "irradiation_poa": poa if pd.notna(poa) else None,
                        "air_temp": row[col_air] if col_air else None,
                        "pv_temp": row[col_pv] if col_pv else None,
                        "power_kw": row[col_power],
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
