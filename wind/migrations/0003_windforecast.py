from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ("stations", "0016_station_station_kind"),
        ("wind", "0002_windrecord"),
    ]

    operations = [
        migrations.CreateModel(
            name="WindForecast",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("timestamp", models.DateTimeField()),
                ("forecast_scope", models.CharField(choices=[("main", "Основная база"), ("test", "Тестовая база")], default="main", max_length=16)),
                ("pred_heur", models.FloatField(blank=True, null=True)),
                ("pred_final", models.FloatField(blank=True, null=True)),
                ("weather_source", models.CharField(blank=True, default="", max_length=32)),
                ("air_temp_fc", models.FloatField(blank=True, null=True)),
                ("wind_speed_fc", models.FloatField(blank=True, null=True)),
                ("wind_direction_fc", models.FloatField(blank=True, null=True)),
                ("cloudcover_fc", models.FloatField(blank=True, null=True)),
                ("humidity_fc", models.FloatField(blank=True, null=True)),
                ("precip_fc", models.FloatField(blank=True, null=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("station", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="wind_forecasts", to="stations.station")),
            ],
            options={
                "ordering": ["timestamp"],
                "indexes": [
                    models.Index(fields=["station", "forecast_scope", "timestamp"], name="wind_windfo_station_a76e39_idx"),
                    models.Index(fields=["station", "timestamp"], name="wind_windfo_station_1912bf_idx"),
                ],
                "unique_together": {("station", "forecast_scope", "timestamp")},
            },
        ),
    ]
