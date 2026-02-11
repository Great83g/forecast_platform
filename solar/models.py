from django.db import models
from stations.models import Station

# Совместимость: в некоторых местах проект ожидал класс SolarStation.
# Чтобы не падать с ImportError, экспортируем Station под этим именем.
SolarStation = Station


class SolarRecord(models.Model):
    HISTORY_SCOPE_MAIN = "main"
    HISTORY_SCOPE_TEST = "test"
    HISTORY_SCOPE_CHOICES = [
        (HISTORY_SCOPE_MAIN, "Основная база"),
        (HISTORY_SCOPE_TEST, "Тестовая база"),
    ]

    """
    Исторические данные станции:
    - timestamp: момент времени
    - irradiation: солнечная радиация (Вт/м²)
    - air_temp: температура воздуха (°C)
    - pv_temp: температура панелей (°C)
    - power_kw: фактическая выработка (кВт)
    """
    station = models.ForeignKey(
        Station,
        on_delete=models.CASCADE,
        related_name="records",
    )
    timestamp = models.DateTimeField()
    history_scope = models.CharField(max_length=16, choices=HISTORY_SCOPE_CHOICES, default=HISTORY_SCOPE_MAIN)

    irradiation = models.FloatField(null=True, blank=True)
    air_temp = models.FloatField(null=True, blank=True)
    pv_temp = models.FloatField(null=True, blank=True)
    power_kw = models.FloatField(null=True, blank=True)

    class Meta:
        ordering = ["timestamp"]
        indexes = [
            models.Index(fields=["station", "history_scope", "timestamp"]),
            models.Index(fields=["station", "timestamp"]),
        ]

    def __str__(self):
        return f"{self.station.name} [{self.history_scope}] @ {self.timestamp}"


class SolarForecast(models.Model):
    SCOPE_MAIN = "main"
    SCOPE_TEST = "test"
    SCOPE_CHOICES = [
        (SCOPE_MAIN, "Основная база"),
        (SCOPE_TEST, "Тестовая база"),
    ]

    """
    Прогноз выработки станции на конкретный час:
    - pred_np: прогноз NeuralProphet (может быть None)
    - pred_xgb: прогноз XGBoost (может быть None)
    - pred_heur: эвристика
    - pred_final: итоговый ансамбль

    Дополнительно сохраняем прогноз погоды:
    - irradiation_fc, air_temp_fc, wind_speed_fc, cloudcover_fc, humidity_fc, precip_fc
    """
    station = models.ForeignKey(
        Station,
        on_delete=models.CASCADE,
        related_name="forecasts",
    )
    timestamp = models.DateTimeField()
    forecast_scope = models.CharField(max_length=16, choices=SCOPE_CHOICES, default=SCOPE_MAIN)

    # прогноз генерации (кВт)
    pred_np = models.FloatField(null=True, blank=True)
    pred_xgb = models.FloatField(null=True, blank=True)
    pred_heur = models.FloatField(null=True, blank=True)
    pred_final = models.FloatField(null=True, blank=True)

    # прогноз погоды из Visual Crossing / заглушка
    irradiation_fc = models.FloatField(null=True, blank=True)
    air_temp_fc = models.FloatField(null=True, blank=True)
    wind_speed_fc = models.FloatField(null=True, blank=True)
    cloudcover_fc = models.FloatField(null=True, blank=True)
    humidity_fc = models.FloatField(null=True, blank=True)
    precip_fc = models.FloatField(null=True, blank=True)
    snowfall_fc = models.FloatField(null=True, blank=True)
    snowdepth_fc = models.FloatField(null=True, blank=True)
    weather_code_fc = models.IntegerField(null=True, blank=True)

    auto_snow_flag = models.IntegerField(null=True, blank=True)
    auto_fog_flag = models.IntegerField(null=True, blank=True)
    auto_winter_factor = models.FloatField(null=True, blank=True)
    manual_snow_factor = models.FloatField(null=True, blank=True)
    winter_factor_applied = models.FloatField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["timestamp"]
        indexes = [
            models.Index(fields=["station", "forecast_scope", "timestamp"]),
            models.Index(fields=["station", "timestamp"]),
        ]
        unique_together = ("station", "forecast_scope", "timestamp")

    def __str__(self):
        return f"Forecast {self.station.name} [{self.forecast_scope}] @ {self.timestamp}"
