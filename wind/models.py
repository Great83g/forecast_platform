from django.core.validators import MinValueValidator
from django.db import models

from stations.models import Station


class WindStationProfile(models.Model):
    station = models.OneToOneField(Station, on_delete=models.CASCADE, related_name="wind_profile")
    turbine_count = models.PositiveIntegerField(default=1, validators=[MinValueValidator(1)])
    turbine_rated_power_kw = models.FloatField(default=3000.0, validators=[MinValueValidator(1.0)])
    hub_height_m = models.FloatField(default=100.0, validators=[MinValueValidator(1.0)])
    rotor_diameter_m = models.FloatField(default=120.0, validators=[MinValueValidator(1.0)])
    cut_in_speed_ms = models.FloatField(default=3.0, validators=[MinValueValidator(0.0)])
    rated_speed_ms = models.FloatField(default=12.0, validators=[MinValueValidator(0.0)])
    cut_out_speed_ms = models.FloatField(default=25.0, validators=[MinValueValidator(0.0)])

    @property
    def installed_capacity_kw(self) -> float:
        return float(self.turbine_count) * float(self.turbine_rated_power_kw)

    def __str__(self) -> str:
        return f"Wind profile for {self.station.name}"


class WindRecord(models.Model):
    HISTORY_SCOPE_MAIN = "main"
    HISTORY_SCOPE_TEST = "test"
    HISTORY_SCOPE_CHOICES = [
        (HISTORY_SCOPE_MAIN, "Основная база"),
        (HISTORY_SCOPE_TEST, "Тестовая база"),
    ]

    station = models.ForeignKey(Station, on_delete=models.CASCADE, related_name="wind_records")
    timestamp = models.DateTimeField()
    history_scope = models.CharField(max_length=16, choices=HISTORY_SCOPE_CHOICES, default=HISTORY_SCOPE_MAIN)

    power_kw = models.FloatField(null=True, blank=True)
    wind_speed_ms = models.FloatField(null=True, blank=True)
    wind_direction_deg = models.FloatField(null=True, blank=True)
    air_temp = models.FloatField(null=True, blank=True)
    air_density = models.FloatField(null=True, blank=True)

    class Meta:
        ordering = ["timestamp"]
        indexes = [
            models.Index(fields=["station", "history_scope", "timestamp"], name="wind_windre_station_a3073c_idx"),
            models.Index(fields=["station", "timestamp"], name="wind_windre_station_9dbbbe_idx"),
        ]

    def __str__(self) -> str:
        return f"Wind history {self.station.name} [{self.history_scope}] @ {self.timestamp}"


class WindForecast(models.Model):
    SCOPE_MAIN = "main"
    SCOPE_TEST = "test"
    SCOPE_CHOICES = [
        (SCOPE_MAIN, "Основная база"),
        (SCOPE_TEST, "Тестовая база"),
    ]

    station = models.ForeignKey(Station, on_delete=models.CASCADE, related_name="wind_forecasts")
    timestamp = models.DateTimeField()
    forecast_scope = models.CharField(max_length=16, choices=SCOPE_CHOICES, default=SCOPE_MAIN)

    pred_heur = models.FloatField(null=True, blank=True)
    pred_final = models.FloatField(null=True, blank=True)

    weather_source = models.CharField(max_length=32, blank=True, default="")
    air_temp_fc = models.FloatField(null=True, blank=True)
    wind_speed_fc = models.FloatField(null=True, blank=True)
    wind_direction_fc = models.FloatField(null=True, blank=True)
    cloudcover_fc = models.FloatField(null=True, blank=True)
    humidity_fc = models.FloatField(null=True, blank=True)
    precip_fc = models.FloatField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["timestamp"]
        indexes = [
            models.Index(fields=["station", "forecast_scope", "timestamp"], name="wind_windfo_station_09a39e_idx"),
            models.Index(fields=["station", "timestamp"], name="wind_windfo_station_cea399_idx"),
        ]
        unique_together = ("station", "forecast_scope", "timestamp")

    def __str__(self) -> str:
        return f"Wind forecast {self.station.name} [{self.forecast_scope}] @ {self.timestamp}"
