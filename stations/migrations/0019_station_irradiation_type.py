# Generated manually for irradiation source-type support.

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("stations", "0018_station_forecast_shift_hours"),
    ]

    operations = [
        migrations.AddField(
            model_name="station",
            name="irradiation_type",
            field=models.CharField(
                choices=[("GHI", "GHI (горизонтальная)"), ("POA", "POA (в плоскости панелей)")],
                default="GHI",
                help_text="Тип старой колонки irradiation, если в истории нет отдельных GHI/POA.",
                max_length=8,
            ),
        ),
    ]
