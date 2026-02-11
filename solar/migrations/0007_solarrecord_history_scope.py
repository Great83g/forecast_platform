from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("solar", "0006_solarforecast_scope"),
    ]

    operations = [
        migrations.AddField(
            model_name="solarrecord",
            name="history_scope",
            field=models.CharField(
                choices=[("main", "Основная база"), ("test", "Тестовая база")],
                default="main",
                max_length=16,
            ),
        ),
        migrations.AddIndex(
            model_name="solarrecord",
            index=models.Index(fields=["station", "history_scope", "timestamp"], name="solar_solar_station_11d137_idx"),
        ),
    ]
