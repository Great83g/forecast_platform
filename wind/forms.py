from django import forms
from django.db.models import Q

from stations.models import Organization, Station
from .models import WindStationProfile


class WindStationForm(forms.ModelForm):
    class Meta:
        model = Station
        fields = ["name", "org", "latitude", "longitude", "timezone", "data_shift_hours"]

    def __init__(self, *args, **kwargs):
        user = kwargs.pop("user", None)
        super().__init__(*args, **kwargs)
        self.fields["name"].label = "Название ветростанции"
        self.fields["org"].label = "Оператор"
        self.fields["latitude"].label = "Широта"
        self.fields["longitude"].label = "Долгота"
        self.fields["timezone"].label = "Часовой пояс"
        self.fields["data_shift_hours"].label = "Сдвиг данных (часы)"
        self.fields["data_shift_hours"].required = False

        if user is not None:
            self.fields["org"].queryset = Organization.objects.filter(
                Q(owner=user) | Q(memberships__user=user)
            ).distinct()


class WindStationProfileForm(forms.ModelForm):
    class Meta:
        model = WindStationProfile
        fields = [
            "turbine_count",
            "turbine_rated_power_kw",
            "hub_height_m",
            "rotor_diameter_m",
            "cut_in_speed_ms",
            "rated_speed_ms",
            "cut_out_speed_ms",
        ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["turbine_count"].label = "Количество турбин"
        self.fields["turbine_rated_power_kw"].label = "Мощность 1 турбины (кВт)"
        self.fields["hub_height_m"].label = "Высота башни (м)"
        self.fields["rotor_diameter_m"].label = "Диаметр ротора (м)"
        self.fields["cut_in_speed_ms"].label = "Скорость включения (м/с)"
        self.fields["rated_speed_ms"].label = "Номинальная скорость (м/с)"
        self.fields["cut_out_speed_ms"].label = "Скорость отключения (м/с)"

    def clean(self):
        cleaned = super().clean()
        cut_in = cleaned.get("cut_in_speed_ms")
        rated = cleaned.get("rated_speed_ms")
        cut_out = cleaned.get("cut_out_speed_ms")

        if cut_in is not None and rated is not None and cut_in >= rated:
            self.add_error("rated_speed_ms", "Номинальная скорость должна быть выше скорости включения.")
        if rated is not None and cut_out is not None and rated >= cut_out:
            self.add_error("cut_out_speed_ms", "Скорость отключения должна быть выше номинальной.")

        return cleaned


class WindForecastScheduleForm(forms.Form):
    enabled = forms.BooleanField(label="Авто‑прогноз", required=False)
    run_time = forms.TimeField(
        label="Время запуска",
        widget=forms.TimeInput(attrs={"type": "time", "class": "form-control form-control-sm"}),
    )
    days = forms.IntegerField(
        label="Дней вперёд",
        min_value=1,
        max_value=7,
        widget=forms.NumberInput(attrs={"class": "form-control form-control-sm", "style": "width: 90px;"}),
    )
    providers = forms.MultipleChoiceField(
        label="Провайдеры",
        required=False,
        widget=forms.CheckboxSelectMultiple,
        choices=[
            ("visual_crossing", "Visual Crossing"),
            ("open_meteo", "Open‑Meteo"),
        ],
    )
    emails = forms.CharField(
        label="Email получателей",
        required=False,
        widget=forms.TextInput(
            attrs={
                "placeholder": "mail1@example.com, mail2@example.com",
                "class": "form-control form-control-sm",
                "style": "width: 320px;",
            }
        ),
    )
    auto_send = forms.BooleanField(label="Авто‑отправка email", required=False)

    scope = forms.ChoiceField(
        label="База прогноза",
        choices=[("main", "Основная"), ("test", "Тестовая")],
        initial="test",
        widget=forms.Select(attrs={"class": "form-select form-select-sm", "style": "width: 160px;"}),
    )
