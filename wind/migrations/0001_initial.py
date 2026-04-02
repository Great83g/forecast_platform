from django.db import migrations, models
import django.core.validators
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ("stations", "0016_station_station_kind"),
    ]

    operations = [
        migrations.CreateModel(
            name="WindStationProfile",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("turbine_count", models.PositiveIntegerField(default=1, validators=[django.core.validators.MinValueValidator(1)])),
                ("turbine_rated_power_kw", models.FloatField(default=3000.0, validators=[django.core.validators.MinValueValidator(1.0)])),
                ("hub_height_m", models.FloatField(default=100.0, validators=[django.core.validators.MinValueValidator(1.0)])),
                ("rotor_diameter_m", models.FloatField(default=120.0, validators=[django.core.validators.MinValueValidator(1.0)])),
                ("cut_in_speed_ms", models.FloatField(default=3.0, validators=[django.core.validators.MinValueValidator(0.0)])),
                ("rated_speed_ms", models.FloatField(default=12.0, validators=[django.core.validators.MinValueValidator(0.0)])),
                ("cut_out_speed_ms", models.FloatField(default=25.0, validators=[django.core.validators.MinValueValidator(0.0)])),
                (
                    "station",
                    models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name="wind_profile", to="stations.station"),
                ),
            ],
        ),
    ]
