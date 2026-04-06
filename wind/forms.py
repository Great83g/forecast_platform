from django import forms
from django.db.models import Q

from stations.models import Organization, Station
from .models import WindStationProfile

AUTO_HISTORY_TIME_INPUT_FORMATS = [
    "%H:%M",
    "%H:%M:%S",
    "%I:%M %p",
    "%I:%M:%S %p",
    "%I:%M%p",
    "%I:%M:%S%p",
]


class WindStationForm(forms.ModelForm):
    class Meta:
        model = Station
        fields = [
            "name",
            "org",
            "latitude",
            "longitude",
            "timezone",
            "data_shift_hours",
            "auto_history_enabled",
            "auto_history_folder",
            "auto_history_script",
            "auto_history_run_time",
        ]

    def __init__(self, *args, **kwargs):
        user = kwargs.pop("user", None)
        super().__init__(*args, **kwargs)
        self.fields["name"].label = "Название ветростанции"
        self.fields["org"].label = "Оператор"
        self.fields["latitude"].label = "Широта"
        self.fields["longitude"].label = "Долгота"
        self.fields["timezone"].label = "Часовой пояс"
        self.fields["data_shift_hours"].label = "Сдвиг данных (часы)"
        self.fields["auto_history_enabled"].label = "Автозаполнение истории"
        self.fields["auto_history_folder"].label = "Папка автоимпорта"
        self.fields["auto_history_script"].label = "Скрипт автоистории"
        self.fields["auto_history_run_time"].label = "Время автопроверки"
        self.fields["data_shift_hours"].required = False
        self.fields["auto_history_enabled"].help_text = (
            "Если включено — планировщик будет автоматически подтягивать историю из указанной папки."
        )
        self.fields["auto_history_folder"].help_text = (
            "Например: /mnt/share или /mnt/share/wind для CSV/XLSX истории ветростанции."
        )
        self.fields["auto_history_script"].help_text = (
            "Куда добавлять: wind/services/history_scripts/. "
            "Формат: module_name, python.module:function_name или /path/to/file.py:function_name."
        )
        self.fields["auto_history_script"].widget.attrs.setdefault(
            "placeholder",
            "example_wind  или  wind.services.history_scripts.example_wind:build_history_dataframe",
        )
        self.fields["auto_history_run_time"].help_text = (
            "Ежедневно в это время будет запускаться проверка новой истории."
        )
        self.fields["auto_history_run_time"].input_formats = AUTO_HISTORY_TIME_INPUT_FORMATS
        self.fields["auto_history_run_time"].widget = forms.TimeInput(attrs={"type": "time", "step": "60"})

        if not self.instance.pk and not self.is_bound:
            self.fields["auto_history_folder"].initial = "/mnt/share/wind"

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
    horizon_mode = forms.ChoiceField(
        label="Режим горизонта",
        required=False,
        choices=[
            ("legacy", "Обычный (старый)"),
            ("weekday_calendar", "Календарь: Пн–Чт → +2 дня, Пт → +2/+3/+4 (Вс/Пн/Вт)"),
        ],
        widget=forms.Select(attrs={"class": "form-select form-select-sm", "style": "width: 360px;"}),
        initial="weekday_calendar",
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
