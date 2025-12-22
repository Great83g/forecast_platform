from django.db import models
from stations.models import Station

# Совместимость: в некоторых местах проект ожидал класс SolarStation.
# Чтобы не падать с ImportError, экспортируем Station под этим именем.
SolarStation = Station


class SolarRecord(models.Model):
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

    irradiation = models.FloatField(null=True, blank=True)
    air_temp = models.FloatField(null=True, blank=True)
    pv_temp = models.FloatField(null=True, blank=True)
    power_kw = models.FloatField(null=True, blank=True)

    class Meta:
        ordering = ["timestamp"]
        indexes = [
            models.Index(fields=["station", "timestamp"]),
        ]

    def __str__(self):
        return f"{self.station.name} @ {self.timestamp}"


class SolarForecast(models.Model):
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

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["timestamp"]
        indexes = [
            models.Index(fields=["station", "timestamp"]),
        ]
        unique_together = ("station", "timestamp")

    def __str__(self):
        return f"Forecast {self.station.name} @ {self.timestamp}"

