# solar/models.py
from django.db import models
from stations.models import Station


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


# ========= ПРОГНОЗЫ ==========

class SolarForecast(models.Model):
    """
    Прогноз выработки станции на конкретный час:
    - pred_np: прогноз NeuralProphet
    - pred_xgb: прогноз XGBoost
    - pred_heur: простая эвристика
    - pred_final: итоговый ансамбль (то, что показываем на графике)
    """
    station = models.ForeignKey(
        Station,
        on_delete=models.CASCADE,
        related_name="forecasts",
    )
    timestamp = models.DateTimeField()

    pred_np = models.FloatField()
    pred_xgb = models.FloatField()
    pred_heur = models.FloatField()
    pred_final = models.FloatField()

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["timestamp"]
        indexes = [
            models.Index(fields=["station", "timestamp"]),
        ]
        unique_together = ("station", "timestamp")

    def __str__(self):
        return f"Forecast {self.station.name} @ {self.timestamp}"

