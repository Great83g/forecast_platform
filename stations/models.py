from django.db import models
from django.contrib.auth.models import User


class Organization(models.Model):
    name = models.CharField(max_length=200)
    owner = models.ForeignKey(User, on_delete=models.CASCADE, related_name="organizations")

    def __str__(self):
        return self.name


class Station(models.Model):
    org = models.ForeignKey(Organization, on_delete=models.CASCADE, related_name="stations")
    name = models.CharField(max_length=200)
    capacity_mw = models.FloatField(default=1.0)
    latitude = models.FloatField(null=True, blank=True)
    longitude = models.FloatField(null=True, blank=True)
    timezone = models.CharField(max_length=100, default="Asia/Almaty")

    def __str__(self):
        return f"{self.name} ({self.capacity_mw} MW)"


# Create your models here.
