from django.db import models

from stations.models import Station


class ForecastReport(models.Model):
    station = models.ForeignKey(Station, on_delete=models.CASCADE, related_name="forecast_reports")
    created_at = models.DateTimeField(auto_now_add=True)
    days = models.PositiveIntegerField(default=1)
    weather_source = models.CharField(max_length=64, blank=True)
    recipients = models.TextField(blank=True)
    file = models.FileField(upload_to="forecast_reports/")
    error = models.TextField(blank=True)

    def __str__(self) -> str:
        return f"ForecastReport(station={self.station_id}, created_at={self.created_at:%Y-%m-%d %H:%M})"
