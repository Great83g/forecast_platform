from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models

from stations.models import Station


class StationBalanceConfig(models.Model):
    MODE_FOLLOW_SCHEDULE = "follow_schedule"
    MODE_PEAK_SHAVING = "peak_shaving"
    MODE_SIMULATION_ONLY = "simulation_only"
    BALANCE_MODE_CHOICES = [
        (MODE_FOLLOW_SCHEDULE, "Follow schedule"),
        (MODE_PEAK_SHAVING, "Peak shaving"),
        (MODE_SIMULATION_ONLY, "Simulation only"),
    ]

    DATA_SOURCE_HISTORY = "history"
    DATA_SOURCE_FUSIONSOLAR = "fusionsolar"
    DATA_SOURCE_MODBUS = "modbus"
    DATA_SOURCE_MANUAL = "manual"
    DATA_SOURCE_CHOICES = [
        (DATA_SOURCE_HISTORY, "History"),
        (DATA_SOURCE_FUSIONSOLAR, "FusionSolar"),
        (DATA_SOURCE_MODBUS, "Modbus"),
        (DATA_SOURCE_MANUAL, "Manual"),
    ]

    station = models.ForeignKey(Station, on_delete=models.CASCADE, related_name="balance_configs")
    enabled = models.BooleanField(default=False)
    station_capacity_mw = models.DecimalField(max_digits=10, decimal_places=3, default=0)
    grid_limit_mw = models.DecimalField(max_digits=10, decimal_places=3, default=0)
    balance_mode = models.CharField(max_length=32, choices=BALANCE_MODE_CHOICES, default=MODE_SIMULATION_ONLY)
    data_source = models.CharField(max_length=32, choices=DATA_SOURCE_CHOICES, default=DATA_SOURCE_HISTORY)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["station__name", "id"]
        verbose_name = "Station balance config"
        verbose_name_plural = "Station balance configs"

    def __str__(self):
        return f"{self.station.name} balance ({self.get_balance_mode_display()})"


class VirtualESSConfig(models.Model):
    TIMESTEP_CHOICES = [
        (60, "60 минут"),
        (15, "15 минут"),
        (5, "5 минут"),
    ]

    station = models.ForeignKey(Station, on_delete=models.CASCADE, related_name="virtual_ess_configs")
    name = models.CharField(max_length=200, default="Virtual ESS")
    enabled = models.BooleanField(default=True)
    power_mw = models.DecimalField(max_digits=10, decimal_places=3, default=0)
    capacity_mwh = models.DecimalField(max_digits=10, decimal_places=3, default=0)
    pcs_power_mw = models.DecimalField(max_digits=10, decimal_places=3, default=0)
    soc_initial_percent = models.DecimalField(
        max_digits=5,
        decimal_places=2,
        default=50,
        validators=[MinValueValidator(0), MaxValueValidator(100)],
    )
    soc_min_percent = models.DecimalField(
        max_digits=5,
        decimal_places=2,
        default=10,
        validators=[MinValueValidator(0), MaxValueValidator(100)],
    )
    soc_max_percent = models.DecimalField(
        max_digits=5,
        decimal_places=2,
        default=90,
        validators=[MinValueValidator(0), MaxValueValidator(100)],
    )
    charge_efficiency = models.DecimalField(
        max_digits=4,
        decimal_places=3,
        default=0.950,
        validators=[MinValueValidator(0), MaxValueValidator(1)],
    )
    discharge_efficiency = models.DecimalField(
        max_digits=4,
        decimal_places=3,
        default=0.950,
        validators=[MinValueValidator(0), MaxValueValidator(1)],
    )
    timestep_minutes = models.PositiveSmallIntegerField(choices=TIMESTEP_CHOICES, default=60)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["station__name", "name", "id"]
        verbose_name = "Virtual ESS config"
        verbose_name_plural = "Virtual ESS configs"

    def __str__(self):
        return f"{self.name} — {self.station.name}"


class ESSSimulationRun(models.Model):
    TYPE_POSTFACTUM = "postfactum"
    TYPE_ONLINE = "online"
    SIMULATION_TYPE_CHOICES = [
        (TYPE_POSTFACTUM, "Postfactum"),
        (TYPE_ONLINE, "Online demo"),
    ]

    STATUS_CREATED = "created"
    STATUS_RUNNING = "running"
    STATUS_FINISHED = "finished"
    STATUS_FAILED = "failed"
    STATUS_CHOICES = [
        (STATUS_CREATED, "Created"),
        (STATUS_RUNNING, "Running"),
        (STATUS_FINISHED, "Finished"),
        (STATUS_FAILED, "Failed"),
    ]

    station = models.ForeignKey(Station, on_delete=models.CASCADE, related_name="ess_simulation_runs")
    ess_config = models.ForeignKey(VirtualESSConfig, on_delete=models.SET_NULL, null=True, blank=True, related_name="simulation_runs")
    simulation_type = models.CharField(max_length=16, choices=SIMULATION_TYPE_CHOICES, default=TYPE_POSTFACTUM)
    date_from = models.DateField()
    date_to = models.DateField()
    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default=STATUS_CREATED)
    created_at = models.DateTimeField(auto_now_add=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at", "-id"]
        verbose_name = "ESS simulation run"
        verbose_name_plural = "ESS simulation runs"

    def __str__(self):
        return f"{self.station.name} {self.get_simulation_type_display()} {self.date_from}–{self.date_to}"


class ESSSimulationPoint(models.Model):
    run = models.ForeignKey(ESSSimulationRun, on_delete=models.CASCADE, related_name="points")
    timestamp = models.DateTimeField(db_index=True)
    plan_mw = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True)
    fact_mw = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True)
    deviation_mw = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True)
    ess_command_mw = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True)
    ess_charge_mw = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True)
    ess_discharge_mw = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True)
    soc_percent = models.DecimalField(max_digits=6, decimal_places=3, null=True, blank=True)
    soc_mwh = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True)
    output_after_ess_mw = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True)
    unbalanced_mw = models.DecimalField(max_digits=12, decimal_places=4, null=True, blank=True)

    class Meta:
        ordering = ["run", "timestamp", "id"]
        verbose_name = "ESS simulation point"
        verbose_name_plural = "ESS simulation points"
        indexes = [models.Index(fields=["run", "timestamp"], name="virtual_ess_run_id_9f84d2_idx")]

    def __str__(self):
        return f"{self.run_id} @ {self.timestamp}"
