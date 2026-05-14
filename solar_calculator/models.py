from django.db import models
from django.db.models import F


class CalculatorUsageCounter(models.Model):
    key = models.CharField(max_length=64, unique=True, default="solar_calculator")
    count = models.PositiveBigIntegerField(default=0)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Solar calculator usage counter"
        verbose_name_plural = "Solar calculator usage counters"

    def __str__(self) -> str:
        return f"{self.key}: {self.count}"

    @classmethod
    def current_count(cls) -> int:
        counter, _ = cls.objects.get_or_create(key="solar_calculator")
        return counter.count

    @classmethod
    def increment(cls) -> int:
        counter, _ = cls.objects.get_or_create(key="solar_calculator")
        cls.objects.filter(pk=counter.pk).update(count=F("count") + 1)
        counter.refresh_from_db(fields=["count"])
        return counter.count
