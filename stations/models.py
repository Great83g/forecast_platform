from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MinValueValidator, MaxValueValidator
from django.core.exceptions import ValidationError


class Organization(models.Model):
    name = models.CharField(max_length=200)
    owner = models.ForeignKey(User, on_delete=models.CASCADE, related_name="organizations")

    def __str__(self):
        return self.name


class Station(models.Model):
    org = models.ForeignKey(Organization, on_delete=models.CASCADE, related_name="stations")
    name = models.CharField(max_length=200)

    # Старое поле оставляем для совместимости с текущим кодом
    capacity_mw = models.FloatField(default=1.0)

    # Координаты (оставляем как есть: latitude/longitude, НЕ добавляем lat/lon чтобы не было путаницы)
    latitude = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(-90.0), MaxValueValidator(90.0)],
    )
    longitude = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(-180.0), MaxValueValidator(180.0)],
    )

    timezone = models.CharField(max_length=100, default="Asia/Almaty")

    # === Паспорт станции (MVP) ===
    # Мощности (кВт)
    capacity_dc_kw = models.FloatField(default=1000.0, validators=[MinValueValidator(0.0)])
    capacity_ac_kw = models.FloatField(default=1000.0, validators=[MinValueValidator(0.0)])

    # PR
    pr_default = models.FloatField(
        default=0.88,
        validators=[MinValueValidator(0.10), MaxValueValidator(1.00)],
    )

    # Геометрия
    tilt_deg = models.FloatField(
        default=30.0,
        validators=[MinValueValidator(0.0), MaxValueValidator(90.0)],
    )
    azimuth_deg = models.FloatField(
        default=180.0,
        validators=[MinValueValidator(0.0), MaxValueValidator(360.0)],
    )

    # Потери (в сумме)
    losses_total_pct = models.FloatField(
        default=10.0,
        validators=[MinValueValidator(0.0), MaxValueValidator(40.0)],
    )

    def clean(self):
        super().clean()
        if (
            self.capacity_ac_kw is not None
            and self.capacity_dc_kw is not None
            and self.capacity_ac_kw > self.capacity_dc_kw
        ):
            raise ValidationError({"capacity_ac_kw": "AC мощность не должна быть больше DC мощности."})

    def __str__(self):
        return f"{self.name} ({self.capacity_mw} MW)"
