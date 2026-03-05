from django.db.models.signals import post_delete, post_save
from django.dispatch import receiver

from solar.models import SolarForecast, SolarRecord
from solar.org_sync import (
    delete_solar_forecast,
    delete_solar_record,
    sync_solar_forecast,
    sync_solar_record,
    sync_station,
)
from stations.models import Station


@receiver(post_save, sender=Station)
def sync_station_to_org_db(sender, instance: Station, **kwargs):
    sync_station(instance)


@receiver(post_save, sender=SolarRecord)
def sync_record_to_org_db(sender, instance: SolarRecord, **kwargs):
    sync_solar_record(instance)


@receiver(post_delete, sender=SolarRecord)
def remove_record_from_org_db(sender, instance: SolarRecord, **kwargs):
    delete_solar_record(instance)


@receiver(post_save, sender=SolarForecast)
def sync_forecast_to_org_db(sender, instance: SolarForecast, **kwargs):
    sync_solar_forecast(instance)


@receiver(post_delete, sender=SolarForecast)
def remove_forecast_from_org_db(sender, instance: SolarForecast, **kwargs):
    delete_solar_forecast(instance)
