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
