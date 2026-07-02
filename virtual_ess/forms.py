from django import forms

from .models import ESSSimulationRun, StationBalanceConfig, VirtualESSConfig


def _apply_bootstrap_widgets(form):
    for field in form.fields.values():
        if isinstance(field.widget, forms.CheckboxInput):
            field.widget.attrs.setdefault("class", "form-check-input")
        elif isinstance(field.widget, forms.Select):
            field.widget.attrs.setdefault("class", "form-select")
        else:
            field.widget.attrs.setdefault("class", "form-control")


class StationBalanceConfigForm(forms.ModelForm):
    class Meta:
        model = StationBalanceConfig
        fields = [
            "enabled",
            "station_capacity_mw",
            "grid_limit_mw",
            "balance_mode",
            "data_source",
        ]
        widgets = {
            "station_capacity_mw": forms.NumberInput(attrs={"step": "0.001"}),
            "grid_limit_mw": forms.NumberInput(attrs={"step": "0.001"}),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["enabled"].label = "Включить балансировку"
        self.fields["station_capacity_mw"].label = "Мощность станции, МВт"
        self.fields["grid_limit_mw"].label = "Лимит выдачи в сеть, МВт"
        self.fields["balance_mode"].label = "Режим балансировки"
        self.fields["data_source"].label = "Источник факта"
        _apply_bootstrap_widgets(self)


class VirtualESSConfigForm(forms.ModelForm):
    class Meta:
        model = VirtualESSConfig
        fields = [
            "name",
            "enabled",
            "power_mw",
            "capacity_mwh",
            "pcs_power_mw",
            "soc_initial_percent",
            "soc_min_percent",
            "soc_max_percent",
            "charge_efficiency",
            "discharge_efficiency",
            "timestep_minutes",
        ]
        widgets = {
            "power_mw": forms.NumberInput(attrs={"step": "0.001"}),
            "capacity_mwh": forms.NumberInput(attrs={"step": "0.001"}),
            "pcs_power_mw": forms.NumberInput(attrs={"step": "0.001"}),
            "soc_initial_percent": forms.NumberInput(attrs={"step": "0.01"}),
            "soc_min_percent": forms.NumberInput(attrs={"step": "0.01"}),
            "soc_max_percent": forms.NumberInput(attrs={"step": "0.01"}),
            "charge_efficiency": forms.NumberInput(attrs={"step": "0.001"}),
            "discharge_efficiency": forms.NumberInput(attrs={"step": "0.001"}),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["name"].label = "Название ESS"
        self.fields["enabled"].label = "ESS включена"
        self.fields["power_mw"].label = "Мощность ESS, МВт"
        self.fields["capacity_mwh"].label = "Емкость ESS, МВт·ч"
        self.fields["pcs_power_mw"].label = "Мощность PCS, МВт"
        self.fields["soc_initial_percent"].label = "Стартовый SOC, %"
        self.fields["soc_min_percent"].label = "Минимальный SOC, %"
        self.fields["soc_max_percent"].label = "Максимальный SOC, %"
        self.fields["charge_efficiency"].label = "КПД заряда"
        self.fields["discharge_efficiency"].label = "КПД разряда"
        self.fields["timestep_minutes"].label = "Шаг расчета, минут"
        _apply_bootstrap_widgets(self)


class ESSSimulationRunForm(forms.ModelForm):
    class Meta:
        model = ESSSimulationRun
        fields = ["date_from", "date_to", "simulation_type"]
        widgets = {
            "date_from": forms.DateInput(attrs={"type": "date"}),
            "date_to": forms.DateInput(attrs={"type": "date"}),
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["date_from"].label = "Дата с"
        self.fields["date_to"].label = "Дата по"
        self.fields["simulation_type"].label = "Тип симуляции"
        _apply_bootstrap_widgets(self)
