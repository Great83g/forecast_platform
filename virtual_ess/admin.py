from django.contrib import admin

from .models import ESSSimulationPoint, ESSSimulationRun, StationBalanceConfig, VirtualESSConfig


@admin.register(StationBalanceConfig)
class StationBalanceConfigAdmin(admin.ModelAdmin):
    list_display = ("id", "station", "enabled", "station_capacity_mw", "grid_limit_mw", "balance_mode", "data_source", "updated_at")
    list_filter = ("enabled", "balance_mode", "data_source")
    search_fields = ("station__name",)


@admin.register(VirtualESSConfig)
class VirtualESSConfigAdmin(admin.ModelAdmin):
    list_display = ("id", "station", "name", "enabled", "power_mw", "capacity_mwh", "pcs_power_mw", "timestep_minutes", "updated_at")
    list_filter = ("enabled", "timestep_minutes")
    search_fields = ("station__name", "name")


class ESSSimulationPointInline(admin.TabularInline):
    model = ESSSimulationPoint
    extra = 0
    readonly_fields = (
        "timestamp",
        "plan_mw",
        "fact_mw",
        "deviation_mw",
        "ess_command_mw",
        "ess_charge_mw",
        "ess_discharge_mw",
        "soc_percent",
        "soc_mwh",
        "output_after_ess_mw",
        "unbalanced_mw",
    )
    can_delete = False


@admin.register(ESSSimulationRun)
class ESSSimulationRunAdmin(admin.ModelAdmin):
    list_display = ("id", "station", "ess_config", "simulation_type", "date_from", "date_to", "status", "created_at", "finished_at")
    list_filter = ("simulation_type", "status")
    search_fields = ("station__name", "ess_config__name")
    inlines = [ESSSimulationPointInline]


@admin.register(ESSSimulationPoint)
class ESSSimulationPointAdmin(admin.ModelAdmin):
    list_display = ("id", "run", "timestamp", "plan_mw", "fact_mw", "ess_command_mw", "soc_percent", "output_after_ess_mw")
    list_filter = ("run__status",)
    search_fields = ("run__station__name", "run__ess_config__name")
