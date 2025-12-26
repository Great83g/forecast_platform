from django import forms
from stations.models import Station


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
        ]

    def __init__(self, *args, **kwargs):
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

        # ---------- ДЕФОЛТЫ (только при создании) ----------
        if not self.instance.pk and not self.is_bound:
            self.fields["capacity_dc_kw"].initial = 1000.0
            self.fields["capacity_ac_kw"].initial = 1000.0
            self.fields["pr_default"].initial = 0.88
            self.fields["tilt_deg"].initial = 30.0
            self.fields["azimuth_deg"].initial = 180.0
            self.fields["losses_total_pct"].initial = 10.0
            self.fields["timezone"].initial = "Asia/Almaty"


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
        max_value=5,
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
            }
        ),
    )
