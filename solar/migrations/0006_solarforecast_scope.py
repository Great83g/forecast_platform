from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("solar", "0005_solarforecast_winter_fields"),
    ]

    operations = [
        migrations.AddField(
            model_name="solarforecast",
            name="forecast_scope",
            field=models.CharField(
                choices=[("main", "Основная база"), ("test", "Тестовая база")],
                default="main",
                max_length=16,
            ),
        ),
        migrations.AlterUniqueTogether(
            name="solarforecast",
            unique_together={("station", "forecast_scope", "timestamp")},
        ),
        migrations.AddIndex(
            model_name="solarforecast",
            index=models.Index(fields=["station", "forecast_scope", "timestamp"], name="solar_solar_station_d55313_idx"),
        ),
    ]
