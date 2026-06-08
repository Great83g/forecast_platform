# Generated manually for split irradiation history columns.

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("solar", "0008_rename_solar_solar_station_d55313_idx_solar_solar_station_4bfd33_idx_and_more"),
    ]

    operations = [
        migrations.AddField(
            model_name="solarrecord",
            name="irradiation_ghi",
            field=models.FloatField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="solarrecord",
            name="irradiation_poa",
            field=models.FloatField(blank=True, null=True),
        ),
    ]
