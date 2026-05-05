from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django.utils import timezone
from datetime import timedelta

from .hcaptcha import verify_hcaptcha
from stations.models import Organization, OrganizationMember


class RegisterForm(UserCreationForm):
    email = forms.EmailField(required=True)
    organization_name = forms.CharField(required=False, max_length=200, label="Название организации")
    hcaptcha_token = forms.CharField(widget=forms.HiddenInput(), required=True)

    class Meta(UserCreationForm.Meta):
        model = User
        fields = ("username", "email", "password1", "password2", "organization_name")

    def __init__(self, *args, remote_ip: str | None = None, **kwargs):
        self.remote_ip = remote_ip
        super().__init__(*args, **kwargs)

    def clean_hcaptcha_token(self):
        token = self.cleaned_data.get("hcaptcha_token", "")
        ok, message = verify_hcaptcha(token=token, remote_ip=self.remote_ip)
        if not ok:
            raise forms.ValidationError(message)
        return token

    def save(self, commit=True):
        user = super().save(commit=False)
        user.email = self.cleaned_data["email"]
        if commit:
            user.save()
            organization_name = self.cleaned_data.get("organization_name", "")
            organization = Organization.objects.create(
                name=organization_name or f"{user.username} Workspace",
                owner=user,
                trial_ends_at=timezone.now() + timedelta(days=7),
            )
            OrganizationMember.objects.create(
                organization=organization,
                user=user,
                role=OrganizationMember.ROLE_OWNER,
            )
        return user
