# Generated manually for Virtual ESS stage 1 scaffold.

import django.core.validators
import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ("stations", "0021_station_tracker_albedo"),
    ]

    operations = [
        migrations.CreateModel(
            name="StationBalanceConfig",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("enabled", models.BooleanField(default=False)),
                ("station_capacity_mw", models.DecimalField(decimal_places=3, default=0, max_digits=10)),
                ("grid_limit_mw", models.DecimalField(decimal_places=3, default=0, max_digits=10)),
                ("balance_mode", models.CharField(choices=[("follow_schedule", "Follow schedule"), ("peak_shaving", "Peak shaving"), ("simulation_only", "Simulation only")], default="simulation_only", max_length=32)),
                ("data_source", models.CharField(choices=[("history", "History"), ("fusionsolar", "FusionSolar"), ("modbus", "Modbus"), ("manual", "Manual")], default="history", max_length=32)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("station", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="balance_configs", to="stations.station")),
            ],
            options={
                "verbose_name": "Station balance config",
                "verbose_name_plural": "Station balance configs",
                "ordering": ["station__name", "id"],
            },
        ),
        migrations.CreateModel(
            name="VirtualESSConfig",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("name", models.CharField(default="Virtual ESS", max_length=200)),
                ("enabled", models.BooleanField(default=True)),
                ("power_mw", models.DecimalField(decimal_places=3, default=0, max_digits=10)),
                ("capacity_mwh", models.DecimalField(decimal_places=3, default=0, max_digits=10)),
                ("pcs_power_mw", models.DecimalField(decimal_places=3, default=0, max_digits=10)),
                ("soc_initial_percent", models.DecimalField(decimal_places=2, default=50, max_digits=5, validators=[django.core.validators.MinValueValidator(0), django.core.validators.MaxValueValidator(100)])),
                ("soc_min_percent", models.DecimalField(decimal_places=2, default=10, max_digits=5, validators=[django.core.validators.MinValueValidator(0), django.core.validators.MaxValueValidator(100)])),
                ("soc_max_percent", models.DecimalField(decimal_places=2, default=90, max_digits=5, validators=[django.core.validators.MinValueValidator(0), django.core.validators.MaxValueValidator(100)])),
                ("charge_efficiency", models.DecimalField(decimal_places=3, default=0.95, max_digits=4, validators=[django.core.validators.MinValueValidator(0), django.core.validators.MaxValueValidator(1)])),
                ("discharge_efficiency", models.DecimalField(decimal_places=3, default=0.95, max_digits=4, validators=[django.core.validators.MinValueValidator(0), django.core.validators.MaxValueValidator(1)])),
                ("timestep_minutes", models.PositiveSmallIntegerField(choices=[(60, "60 минут"), (15, "15 минут"), (5, "5 минут")], default=60)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
                ("station", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="virtual_ess_configs", to="stations.station")),
            ],
            options={
                "verbose_name": "Virtual ESS config",
                "verbose_name_plural": "Virtual ESS configs",
                "ordering": ["station__name", "name", "id"],
            },
        ),
        migrations.CreateModel(
            name="ESSSimulationRun",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("simulation_type", models.CharField(choices=[("postfactum", "Postfactum"), ("online", "Online demo")], default="postfactum", max_length=16)),
                ("date_from", models.DateField()),
                ("date_to", models.DateField()),
                ("status", models.CharField(choices=[("created", "Created"), ("running", "Running"), ("finished", "Finished"), ("failed", "Failed")], default="created", max_length=16)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("finished_at", models.DateTimeField(blank=True, null=True)),
                ("ess_config", models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name="simulation_runs", to="virtual_ess.virtualessconfig")),
                ("station", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="ess_simulation_runs", to="stations.station")),
            ],
            options={
                "verbose_name": "ESS simulation run",
                "verbose_name_plural": "ESS simulation runs",
                "ordering": ["-created_at", "-id"],
            },
        ),
        migrations.CreateModel(
            name="ESSSimulationPoint",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("timestamp", models.DateTimeField(db_index=True)),
                ("plan_mw", models.DecimalField(blank=True, decimal_places=4, max_digits=12, null=True)),
                ("fact_mw", models.DecimalField(blank=True, decimal_places=4, max_digits=12, null=True)),
                ("deviation_mw", models.DecimalField(blank=True, decimal_places=4, max_digits=12, null=True)),
                ("ess_command_mw", models.DecimalField(blank=True, decimal_places=4, max_digits=12, null=True)),
                ("ess_charge_mw", models.DecimalField(blank=True, decimal_places=4, max_digits=12, null=True)),
                ("ess_discharge_mw", models.DecimalField(blank=True, decimal_places=4, max_digits=12, null=True)),
                ("soc_percent", models.DecimalField(blank=True, decimal_places=3, max_digits=6, null=True)),
                ("soc_mwh", models.DecimalField(blank=True, decimal_places=4, max_digits=12, null=True)),
                ("output_after_ess_mw", models.DecimalField(blank=True, decimal_places=4, max_digits=12, null=True)),
                ("unbalanced_mw", models.DecimalField(blank=True, decimal_places=4, max_digits=12, null=True)),
                ("run", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name="points", to="virtual_ess.esssimulationrun")),
            ],
            options={
                "verbose_name": "ESS simulation point",
                "verbose_name_plural": "ESS simulation points",
                "ordering": ["run", "timestamp", "id"],
                "indexes": [models.Index(fields=["run", "timestamp"], name="virtual_ess_run_id_9f84d2_idx")],
            },
        ),
    ]
