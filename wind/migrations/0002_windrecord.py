from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ("stations", "0016_station_station_kind"),
        ("wind", "0001_initial"),
    ]

    operations = [
        migrations.CreateModel(
            name="WindRecord",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("timestamp", models.DateTimeField()),
                (
                    "history_scope",
                    models.CharField(
                        choices=[("main", "Основная база"), ("test", "Тестовая база")],
                        default="main",
                        max_length=16,
                    ),
                ),
                ("power_kw", models.FloatField(blank=True, null=True)),
                ("wind_speed_ms", models.FloatField(blank=True, null=True)),
                ("wind_direction_deg", models.FloatField(blank=True, null=True)),
                ("air_temp", models.FloatField(blank=True, null=True)),
                ("air_density", models.FloatField(blank=True, null=True)),
                (
                    "station",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="wind_records",
                        to="stations.station",
                    ),
                ),
            ],
            options={
                "ordering": ["timestamp"],
                "indexes": [
                    models.Index(fields=["station", "history_scope", "timestamp"], name="wind_windre_station_bf6cd4_idx"),
                    models.Index(fields=["station", "timestamp"], name="wind_windre_station_a47ce9_idx"),
                ],
            },
        ),
    ]
