from django import forms
from django.db.models import Q

from stations.models import Organization, Station



AUTO_HISTORY_TIME_INPUT_FORMATS = [
    "%H:%M",
    "%H:%M:%S",
    "%I:%M %p",
    "%I:%M:%S %p",
    "%I:%M%p",
    "%I:%M:%S%p",
]


def _organizations_for_user(user):
    if user is None:
        return Organization.objects.none()
    return Organization.objects.filter(
        Q(owner=user) | Q(memberships__user=user)
    ).distinct()


class StationForm(forms.ModelForm):
    class Meta:
        model = Station
        fields = [
            "name",
            "org",

            # legacy поле (оставляем, чтобы ничего не сломать)
            "capacity_mw",

            # координаты
            "latitude",
            "longitude",
            "timezone",
            "data_shift_hours",
            "forecast_shift_hours",

            # === Паспорт станции (MVP) ===
            "capacity_dc_kw",
            "capacity_ac_kw",
            "mount_type",
            "tracker_axis_tilt",
            "tracker_axis_azimuth",
            "tracker_max_angle",
            "tracker_gcr",
            "tracker_backtrack",
            "tracker_poa_model",
            "irradiation_type",
            "pr_default",
            "tilt_deg",
            "azimuth_deg",
            "losses_total_pct",

            "history_source",
            "history_scale_by_capacity",
            "auto_history_enabled",
            "auto_history_folder",
            "auto_history_script",
            "auto_history_run_time",
        ]

    def __init__(self, *args, **kwargs):
        user = kwargs.pop("user", None)
        super().__init__(*args, **kwargs)

        # ---------- ЛЕЙБЛЫ ----------
        self.fields["name"].label = "Название станции"
        self.fields["org"].label = "Оператор"

        self.fields["capacity_mw"].label = "Номинал (MW, legacy)"

        self.fields["latitude"].label = "Широта"
        self.fields["longitude"].label = "Долгота"
        self.fields["timezone"].label = "Часовой пояс"
        self.fields["data_shift_hours"].label = "Сдвиг данных (часы)"
        self.fields["forecast_shift_hours"].label = "Сдвиг прогноза (часы)"

        self.fields["capacity_dc_kw"].label = "DC мощность (кВт)"
        self.fields["capacity_ac_kw"].label = "AC мощность (кВт)"
        self.fields["mount_type"].label = "Тип конструкции"
        self.fields["tracker_axis_tilt"].label = "Tracker axis_tilt"
        self.fields["tracker_axis_azimuth"].label = "Tracker axis_azimuth"
        self.fields["tracker_max_angle"].label = "Tracker max_angle"
        self.fields["tracker_gcr"].label = "Tracker GCR"
        self.fields["tracker_backtrack"].label = "Tracker backtrack"
        self.fields["tracker_poa_model"].label = "Tracker pvlib model"
        self.fields["irradiation_type"].label = "Тип старой irradiation"
        self.fields["pr_default"].label = "PR (0–1)"
        self.fields["tilt_deg"].label = "Наклон (°)"
        self.fields["azimuth_deg"].label = "Азимут (°), юг = 180"
        self.fields["losses_total_pct"].label = "Потери (%)"

        self.fields["history_source"].label = "Источник истории"
        self.fields["history_scale_by_capacity"].label = "Масштабировать по мощности"
        self.fields["auto_history_enabled"].label = "Автозаполнение истории"
        self.fields["auto_history_folder"].label = "Папка автоимпорта"
        self.fields["auto_history_script"].label = "Скрипт автоистории"
        self.fields["auto_history_run_time"].label = "Время автопроверки"
        self.fields["history_source"].help_text = (
            "Если у станции нет своей истории, можно выбрать близкую станцию."
        )
        self.fields["history_scale_by_capacity"].help_text = (
            "При включении мощность берётся пропорционально (например 1.2/8.8)."
        )
        self.fields["auto_history_enabled"].help_text = (
            "Если включено — при запуске планировщика история автоматически обновится из папки ниже."
        )
        self.fields["auto_history_folder"].help_text = (
            "Например: /mnt/share (SMB/CIFS шары с D222*.csv.gz и Plant Report/reportSPP .xlsx)."
        )
        self.fields["auto_history_script"].help_text = (
            "Куда добавлять: dashboard/services/history_scripts/. "
            "Формат: module_name (например ses_8_8mw), "
            "или python.module:function_name, "
            "или /path/to/file.py:function_name. "
            "Если пусто — используется стандартный обработчик."
        )
        self.fields["auto_history_script"].widget.attrs.setdefault(
            "placeholder",
            "ses_8_8mw  или  dashboard.services.history_scripts.ses_8_8mw:build_history_dataframe",
        )
        self.fields["auto_history_run_time"].help_text = (
            "Ежедневно в это время станция будет проверяться на новые файлы истории."
        )
        self.fields["data_shift_hours"].help_text = (
            "Единый сдвиг времени для визуального выравнивания и автоистории. "
            "Например, -1 или +1 для выравнивания после смены времени."
        )
        self.fields["forecast_shift_hours"].help_text = (
            "Применяется только к сохраняемому прогнозу. Например -1 сдвигает прогноз на час назад."
        )
        self.fields["tracker_axis_tilt"].help_text = "Для одноосевого трекера: обычно 0."
        self.fields["tracker_axis_azimuth"].help_text = "Для одноосевого трекера: обычно 0 для оси север-юг."
        self.fields["tracker_max_angle"].help_text = "Для одноосевого трекера: по умолчанию 30°."
        self.fields["tracker_gcr"].help_text = "Для одноосевого трекера: по умолчанию 0.40."
        self.fields["tracker_backtrack"].help_text = "Для одноосевого трекера: включать backtracking в pvlib."
        self.fields["tracker_poa_model"].help_text = "Модель sky diffuse pvlib для GHI→POA, по умолчанию perez."
        self.fields["irradiation_type"].help_text = (
            "Используется только для старой колонки irradiation, когда нет отдельных irradiation_ghi/irradiation_poa."
        )
        self.fields["auto_history_run_time"].input_formats = AUTO_HISTORY_TIME_INPUT_FORMATS
        self.fields["auto_history_run_time"].widget = forms.TimeInput(attrs={"type": "time", "step": "60"})
        self.fields["data_shift_hours"].required = False
        self.fields["forecast_shift_hours"].required = False

        # ---------- ДЕФОЛТЫ (только при создании) ----------
        if not self.instance.pk and not self.is_bound:
            self.fields["capacity_dc_kw"].initial = 1000.0
            self.fields["capacity_ac_kw"].initial = 1000.0
            self.fields["pr_default"].initial = 0.88
            self.fields["tilt_deg"].initial = 30.0
            self.fields["azimuth_deg"].initial = 180.0
            self.fields["losses_total_pct"].initial = 10.0
            self.fields["mount_type"].initial = Station.MOUNT_FIXED
            self.fields["tracker_axis_tilt"].initial = 0.0
            self.fields["tracker_axis_azimuth"].initial = 0.0
            self.fields["tracker_max_angle"].initial = 30.0
            self.fields["tracker_gcr"].initial = 0.40
            self.fields["tracker_backtrack"].initial = True
            self.fields["tracker_poa_model"].initial = "perez"
            self.fields["irradiation_type"].initial = Station.IRRADIATION_GHI
            self.fields["timezone"].initial = "Asia/Almaty"
            self.fields["data_shift_hours"].initial = 0
            self.fields["forecast_shift_hours"].initial = 0
            self.fields["auto_history_folder"].initial = "/mnt/share"
        if self.instance.pk and not self.is_bound:
            folder = (self.instance.auto_history_folder or "").rstrip("/")
            if folder == "/mnt/share":
                self.fields["auto_history_folder"].initial = self.instance._build_preferred_auto_history_folder()

        if "org" in self.fields:
            self.fields["org"].queryset = _organizations_for_user(user)

        if "history_source" in self.fields:
            qs = Station.objects.all()
            if user is not None and not user.is_superuser:
                qs = qs.filter(org__memberships__user=user).distinct()
            if self.instance.pk:
                qs = qs.exclude(pk=self.instance.pk)
            self.fields["history_source"].queryset = qs

    def clean(self):
        cleaned_data = super().clean()
        capacity_mw = cleaned_data.get("capacity_mw")
        capacity_ac_kw = cleaned_data.get("capacity_ac_kw")
        if capacity_mw and capacity_ac_kw and capacity_mw > 100:
            cleaned_data["capacity_mw"] = capacity_ac_kw / 1000.0

        if cleaned_data.get("data_shift_hours") in (None, ""):
            cleaned_data["data_shift_hours"] = 0
        if cleaned_data.get("forecast_shift_hours") in (None, ""):
            cleaned_data["forecast_shift_hours"] = 0
        return cleaned_data


class UploadHistoryForm(forms.Form):
    file = forms.FileField(label="CSV / Excel файл с историей")
    ghi_column = forms.CharField(label="GHI column", required=False)
    poa_column = forms.CharField(label="POA column", required=False)
    power_column = forms.CharField(label="Power column", required=False)
    air_temp_column = forms.CharField(label="Air temp column", required=False)
    pv_temp_column = forms.CharField(label="PV temp column", required=False)


class ForecastEmailForm(forms.Form):
    emails = forms.CharField(
        label="Email получателей",
        required=False,
        widget=forms.TextInput(
            attrs={
                "placeholder": "mail1@example.com, mail2@example.com",
                "class": "form-control form-control-sm",
            }
        ),
    )


class ForecastScheduleForm(forms.Form):
    enabled = forms.BooleanField(label="Авто‑прогноз", required=False)
    start_at = forms.DateTimeField(
        label="Старт",
        required=False,
        widget=forms.DateTimeInput(attrs={"type": "datetime-local", "class": "form-control form-control-sm"}),
    )
    run_time = forms.TimeField(
        label="Время запуска",
        widget=forms.TimeInput(attrs={"type": "time", "class": "form-control form-control-sm"}),
    )
    test_enabled = forms.BooleanField(label="Авто в TEST", required=False)
    test_run_time = forms.TimeField(
        label="Время TEST",
        required=False,
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
            ("open_meteo_only", "Open‑Meteo без истории (только эвристика)"),
            ("visual_crossing_only", "Visual Crossing без истории (только эвристика)"),
        ],
    )
    test_providers = forms.MultipleChoiceField(
        label="Провайдеры TEST",
        required=False,
        widget=forms.CheckboxSelectMultiple,
        choices=providers.choices,
    )
    emails = forms.CharField(
        label="Email получателей",
        required=False,
        widget=forms.TextInput(
            attrs={
                "placeholder": "mail1@example.com, mail2@example.com",
                "class": "form-control form-control-sm",
            }
        ),
    )
    manual_snow_enable = forms.BooleanField(label="Снег/облака (ручной фактор)", required=False)
    manual_snow_factor = forms.FloatField(
        label="Снег фактор",
        required=False,
        min_value=0.0,
        max_value=1.5,
        widget=forms.NumberInput(
            attrs={
                "class": "form-control form-control-sm",
                "style": "width: 90px;",
                "step": "0.05",
                "placeholder": "1.10",
                "max": "1.5",
                "title": "< 1 уменьшает прогноз, > 1 увеличивает прогноз (до 1.5)",
            }
        ),
    )
    manual_snow_dates = forms.CharField(
        label="Даты снега",
        required=False,
        widget=forms.TextInput(
            attrs={
                "placeholder": "2025-02-05, 2025-02-06",
                "class": "form-control form-control-sm",
                "style": "width: 160px;",
            }
        ),
    )
