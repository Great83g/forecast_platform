from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import migrations, models


def set_ses_12mw_forecast_shift(apps, schema_editor):
    Station = apps.get_model("stations", "Station")
    Station.objects.filter(name="SES 1.2 MW").update(forecast_shift_hours=-1)


def reset_ses_12mw_forecast_shift(apps, schema_editor):
    Station = apps.get_model("stations", "Station")
    Station.objects.filter(name="SES 1.2 MW").update(forecast_shift_hours=0)


class Migration(migrations.Migration):

    dependencies = [
        ("stations", "0017_station_mount_type"),
    ]

    operations = [
        migrations.AddField(
            model_name="station",
            name="forecast_shift_hours",
            field=models.IntegerField(
                default=0,
                help_text="Сдвиг времени только для сохраняемого прогноза (БД/экспорт).",
                validators=[MinValueValidator(-12), MaxValueValidator(12)],
            ),
        ),
        migrations.RunPython(set_ses_12mw_forecast_shift, reset_ses_12mw_forecast_shift),
    ]
