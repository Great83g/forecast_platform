# dashboard/services/forecast_reports.py
from __future__ import annotations

from io import BytesIO
from typing import Iterable, List, Optional

import pandas as pd
from django.core.files.base import ContentFile
from django.utils import timezone

from solar.models import SolarForecast
from stations.models import Station
from django.core.mail import EmailMessage

from dashboard.models import ForecastReport


def _excel_safe_datetime(series: pd.Series) -> pd.Series:
    s = pd.to_datetime(series, errors="coerce")
    try:
        if getattr(s.dt, "tz", None) is not None:
            s = s.dt.tz_convert(timezone.get_current_timezone())
            s = s.dt.tz_localize(None)
        else:
            s = s.dt.tz_localize(None)
    except Exception:
        pass
    return s


def _normalize_recipients(recipients: Optional[Iterable[str]]) -> List[str]:
    if not recipients:
        return []
    cleaned = []
    for value in recipients:
        if not value:
            continue
        if isinstance(value, str):
            parts = [p.strip() for p in value.replace(";", ",").split(",")]
            cleaned.extend([p for p in parts if p])
        else:
            cleaned.append(str(value).strip())
    return [r for r in cleaned if r]


def _forecast_date_range(days: int) -> tuple:
    now = timezone.localtime(timezone.now())
    start_date = (now + pd.Timedelta(days=1)).date()
    start = timezone.datetime.combine(
        start_date,
        timezone.datetime.min.time(),
    ).replace(tzinfo=now.tzinfo)
    end = start + pd.Timedelta(days=days)
    return start, end


def build_forecast_report(
    station: Station,
    days: int,
    weather_source: str,
    recipients: Optional[Iterable[str]] = None,
) -> ForecastReport:
    start, end = _forecast_date_range(days)
    qs = SolarForecast.objects.filter(
        station=station,
        timestamp__gte=start,
        timestamp__lt=end,
    ).order_by("timestamp")
    data = list(
        qs.values(
            "timestamp",
            "pred_np",
            "pred_xgb",
            "pred_heur",
            "irradiation_fc",
            "air_temp_fc",
            "wind_speed_fc",
            "cloudcover_fc",
            "humidity_fc",
            "precip_fc",
            "pred_final",
        )
    )
    df = pd.DataFrame(data)
    if df.empty:
        df = pd.DataFrame(
            columns=[
                "timestamp",
                "pred_np",
                "pred_xgb",
                "pred_heur",
                "irradiation_fc",
                "air_temp_fc",
                "wind_speed_fc",
                "cloudcover_fc",
                "humidity_fc",
                "precip_fc",
                "pred_final",
            ]
        )

    if "timestamp" in df.columns and not df.empty:
        df["timestamp"] = _excel_safe_datetime(df["timestamp"])

    out = BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="forecast")
    out.seek(0)

    stamp = timezone.localtime(timezone.now()).strftime("%Y%m%d_%H%M%S")
    filename = f"forecast_station_{station.pk}_{stamp}.xlsx"
    report = ForecastReport(
        station=station,
        days=days,
        weather_source=weather_source or "",
        recipients=", ".join(_normalize_recipients(recipients)),
    )
    report.file.save(filename, ContentFile(out.getvalue()), save=False)
    report.save()
    return report


def send_report_email(report: ForecastReport, recipients: Iterable[str], station_name: str, days: int) -> bool:
    cleaned = _normalize_recipients(recipients)
    if not cleaned:
        return False
    try:
        subject = f"Прогноз для {station_name} ({days} дн.)"
        body = f"Отчёт по прогнозу для станции {station_name} во вложении."
        email = EmailMessage(subject=subject, body=body, to=cleaned)
        email.attach_file(report.file.path)
        email.send(fail_silently=False)
        return True
    except Exception as exc:
        report.error = str(exc)
        report.save(update_fields=["error"])
        return False
