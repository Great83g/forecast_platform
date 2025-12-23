from django.db import migrations


def rename_station_forward(apps, schema_editor):
    Station = apps.get_model("stations", "Station")
    Station.objects.filter(name="SES 10MW").update(name="SES 8.8MW")


def rename_station_backward(apps, schema_editor):
    Station = apps.get_model("stations", "Station")
    Station.objects.filter(name="SES 8.8MW").update(name="SES 10MW")


class Migration(migrations.Migration):

    dependencies = [
        ("stations", "0002_station_azimuth_deg_station_capacity_ac_kw_and_more"),
    ]

    operations = [
        migrations.RunPython(rename_station_forward, rename_station_backward),
    ]
