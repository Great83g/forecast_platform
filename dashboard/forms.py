from django import forms

from stations.models import Organization, Station


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

            # === Паспорт станции (MVP) ===
            "capacity_dc_kw",
            "capacity_ac_kw",
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

        self.fields["capacity_dc_kw"].label = "DC мощность (кВт)"
        self.fields["capacity_ac_kw"].label = "AC мощность (кВт)"
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
        self.fields["auto_history_run_time"].widget = forms.TimeInput(attrs={"type": "time"})

        # ---------- ДЕФОЛТЫ (только при создании) ----------
        if not self.instance.pk and not self.is_bound:
            self.fields["capacity_dc_kw"].initial = 1000.0
            self.fields["capacity_ac_kw"].initial = 1000.0
            self.fields["pr_default"].initial = 0.88
            self.fields["tilt_deg"].initial = 30.0
            self.fields["azimuth_deg"].initial = 180.0
            self.fields["losses_total_pct"].initial = 10.0
            self.fields["timezone"].initial = "Asia/Almaty"
            self.fields["auto_history_folder"].initial = "/mnt/share"
        if self.instance.pk and not self.is_bound:
            folder = (self.instance.auto_history_folder or "").rstrip("/")
            if folder == "/mnt/share":
                self.fields["auto_history_folder"].initial = Station._build_auto_history_folder(self.instance.name, self.instance.org_id)

        if "org" in self.fields and user is not None:
            org_qs = Organization.objects.all() if user.is_superuser else Organization.objects.filter(memberships__user=user).distinct()
            self.fields["org"].queryset = org_qs

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
        return cleaned_data


class UploadHistoryForm(forms.Form):
    file = forms.FileField(label="CSV / Excel файл с историей")


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
            ("weekday_calendar", "Календарь: Пн–Чт → +2 дня, Пт → +1/+2/+3"),
        ],
        widget=forms.Select(attrs={"class": "form-select form-select-sm", "style": "width: 290px;"}),
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
        ],
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
        max_value=1.0,
        widget=forms.NumberInput(attrs={"class": "form-control form-control-sm", "style": "width: 90px;"}),
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
