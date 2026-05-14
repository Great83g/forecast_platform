from django.db import migrations, models
import django.db.models.deletion
from django.utils import timezone


def create_runs_for_existing_forecasts(apps, schema_editor):
    WindForecast = apps.get_model("wind", "WindForecast")
    WindForecastRun = apps.get_model("wind", "WindForecastRun")

    groups = {}
    for row in WindForecast.objects.filter(forecast_run__isnull=True).order_by("station_id", "forecast_scope", "created_at", "id"):
        created_at = row.created_at or timezone.now()
        base_date = timezone.localtime(created_at).date() if timezone.is_aware(created_at) else created_at.date()
        key = (row.station_id, row.forecast_scope, row.weather_source or "", base_date, created_at)
        groups.setdefault(key, []).append(row.pk)

    for (station_id, scope, provider, base_date, created_at), row_ids in groups.items():
        dates = set(
            WindForecast.objects.filter(pk__in=row_ids, timestamp__isnull=False)
            .values_list("timestamp__date", flat=True)
        )
        run = WindForecastRun.objects.create(
            station_id=station_id,
            forecast_scope=scope,
            forecast_base_date=base_date,
            provider=provider,
            horizon_days=max(1, len(dates)),
        )
        WindForecastRun.objects.filter(pk=run.pk).update(created_at=created_at)
        WindForecast.objects.filter(pk__in=row_ids).update(forecast_run_id=run.pk)


class Migration(migrations.Migration):

    dependencies = [
        ("wind", "0004_rename_wind_windfo_station_a76e39_idx_wind_windfo_station_09a39e_idx_and_more"),
    ]

    operations = [
        migrations.CreateModel(
            name="WindForecastRun",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("forecast_scope", models.CharField(choices=[("main", "Основная база"), ("test", "Тестовая база")], default="main", max_length=16)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("forecast_base_date", models.DateField()),
                ("provider", models.CharField(blank=True, default="", max_length=64)),
                ("horizon_days", models.PositiveIntegerField(default=1)),
                ("station", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="wind_forecast_runs", to="stations.station")),
            ],
            options={
                "ordering": ["-created_at", "-id"],
                "indexes": [
                    models.Index(fields=["station", "forecast_scope", "forecast_base_date"], name="wind_windru_station_43bcb5_idx"),
                    models.Index(fields=["station", "forecast_scope", "created_at"], name="wind_windru_station_49fbec_idx"),
                ],
            },
        ),
        migrations.AddField(
            model_name="windforecast",
            name="forecast_run",
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name="rows", to="wind.windforecastrun"),
        ),
        migrations.AlterUniqueTogether(
            name="windforecast",
            unique_together=set(),
        ),
        migrations.AddIndex(
            model_name="windforecast",
            index=models.Index(fields=["forecast_run", "timestamp"], name="wind_windfo_forecas_ef59c2_idx"),
        ),
        migrations.RunPython(create_runs_for_existing_forecasts, migrations.RunPython.noop),
        migrations.AlterField(
            model_name="windforecast",
            name="forecast_run",
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="rows", to="wind.windforecastrun"),
        ),
        migrations.AddConstraint(
            model_name="windforecast",
            constraint=models.UniqueConstraint(fields=("forecast_run", "timestamp"), name="wind_forecast_run_timestamp_uniq"),
        ),
    ]
