# dashboard/forms.py

from django import forms
from stations.models import Station


class StationForm(forms.ModelForm):
    class Meta:
        model = Station
        fields = ["name", "capacity_mw", "latitude", "longitude", "timezone", "org"]


class UploadHistoryForm(forms.Form):
    file = forms.FileField(label="CSV/Excel файл с историей")

