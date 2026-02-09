from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("dashboard", "0003_forecastschedule_start_at"),
    ]

    operations = [
        migrations.AddField(
            model_name="forecastschedule",
            name="manual_snow_enable",
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name="forecastschedule",
            name="manual_snow_factor",
            field=models.FloatField(default=1.0),
        ),
        migrations.AddField(
            model_name="forecastschedule",
            name="manual_snow_dates",
            field=models.TextField(blank=True),
        ),
    ]
