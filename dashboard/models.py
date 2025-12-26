from datetime import time

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


class ForecastSchedule(models.Model):
    station = models.ForeignKey(Station, on_delete=models.CASCADE, related_name="forecast_schedules")
    enabled = models.BooleanField(default=False)
    run_time = models.TimeField(default=time(6, 0))
    days = models.PositiveIntegerField(default=1)
    providers = models.CharField(max_length=128, blank=True)
    emails = models.TextField(blank=True)
    last_run_at = models.DateTimeField(null=True, blank=True)

    def __str__(self) -> str:
        return f"ForecastSchedule(station={self.station_id}, run_time={self.run_time})"
