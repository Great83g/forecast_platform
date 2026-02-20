import secrets

from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MinValueValidator, MaxValueValidator
from django.core.exceptions import ValidationError
from django.utils import timezone


class Organization(models.Model):
    SUBSCRIPTION_TRIALING = "trialing"
    SUBSCRIPTION_ACTIVE = "active"
    SUBSCRIPTION_PAST_DUE = "past_due"
    SUBSCRIPTION_CANCELED = "canceled"
    SUBSCRIPTION_CHOICES = [
        (SUBSCRIPTION_TRIALING, "Trialing"),
        (SUBSCRIPTION_ACTIVE, "Active"),
        (SUBSCRIPTION_PAST_DUE, "Past due"),
        (SUBSCRIPTION_CANCELED, "Canceled"),
    ]

    name = models.CharField(max_length=200)
    owner = models.ForeignKey(User, on_delete=models.CASCADE, related_name="organizations")
    created_at = models.DateTimeField(auto_now_add=True)
    trial_ends_at = models.DateTimeField(null=True, blank=True)
    is_active = models.BooleanField(default=True)
    subscription_status = models.CharField(
        max_length=16,
        choices=SUBSCRIPTION_CHOICES,
        default=SUBSCRIPTION_TRIALING,
    )

    def is_trial_active(self) -> bool:
        return bool(self.trial_ends_at and self.trial_ends_at >= timezone.now())

    def can_write(self) -> bool:
        if not self.is_active:
            return False
        if self.subscription_status == self.SUBSCRIPTION_ACTIVE:
            return True
        if self.subscription_status == self.SUBSCRIPTION_TRIALING:
            return self.is_trial_active()
        return False

    def write_access_reason(self) -> str:
        if not self.is_active:
            return "Организация деактивирована."
        if self.subscription_status == self.SUBSCRIPTION_ACTIVE:
            return ""
        if self.subscription_status == self.SUBSCRIPTION_TRIALING and not self.is_trial_active():
            return "Пробный период завершён. Обновите тариф, чтобы продолжить запись данных."
        if self.subscription_status == self.SUBSCRIPTION_PAST_DUE:
            return "Есть задолженность по подписке. Погасите счёт, чтобы разблокировать запись данных."
        if self.subscription_status == self.SUBSCRIPTION_CANCELED:
            return "Подписка отменена. Обновите тариф, чтобы продолжить запись данных."
        return "Запись данных недоступна для текущего состояния подписки."

    def __str__(self):
        return self.name


class OrganizationMember(models.Model):
    ROLE_OWNER = "owner"
    ROLE_ADMIN = "admin"
    ROLE_ANALYST = "analyst"
    ROLE_VIEWER = "viewer"
    ROLE_CHOICES = [
        (ROLE_OWNER, "Owner"),
        (ROLE_ADMIN, "Admin"),
        (ROLE_ANALYST, "Analyst"),
        (ROLE_VIEWER, "Viewer"),
    ]

    organization = models.ForeignKey(Organization, on_delete=models.CASCADE, related_name="memberships")
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="organization_memberships")
    role = models.CharField(max_length=16, choices=ROLE_CHOICES, default=ROLE_VIEWER)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("organization", "user")

    def __str__(self):
        return f"{self.user.username} -> {self.organization.name} ({self.role})"




class OrganizationInvitation(models.Model):
    STATUS_PENDING = "pending"
    STATUS_ACCEPTED = "accepted"
    STATUS_CANCELLED = "cancelled"
    STATUS_CHOICES = [
        (STATUS_PENDING, "Pending"),
        (STATUS_ACCEPTED, "Accepted"),
        (STATUS_CANCELLED, "Cancelled"),
    ]

    organization = models.ForeignKey(Organization, on_delete=models.CASCADE, related_name="invitations")
    invited_email = models.EmailField()
    role = models.CharField(max_length=16, choices=OrganizationMember.ROLE_CHOICES, default=OrganizationMember.ROLE_VIEWER)
    token = models.CharField(max_length=64, unique=True, db_index=True)
    status = models.CharField(max_length=16, choices=STATUS_CHOICES, default=STATUS_PENDING)
    invited_by = models.ForeignKey(User, on_delete=models.CASCADE, related_name="sent_org_invitations")
    accepted_by = models.ForeignKey(
        User,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="accepted_org_invitations",
    )
    expires_at = models.DateTimeField()
    accepted_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=["organization", "status"]),
            models.Index(fields=["invited_email", "status"]),
        ]

    def save(self, *args, **kwargs):
        if not self.token:
            self.token = secrets.token_urlsafe(32)
        super().save(*args, **kwargs)

    @property
    def is_expired(self) -> bool:
        return self.expires_at < timezone.now()

    def __str__(self):
        return f"Invite {self.invited_email} -> {self.organization.name} ({self.status})"


class Station(models.Model):
    org = models.ForeignKey(Organization, on_delete=models.CASCADE, related_name="stations")
    name = models.CharField(max_length=200)

    # Старое поле оставляем для совместимости с текущим кодом
    capacity_mw = models.FloatField(default=1.0)

    # Координаты (оставляем как есть: latitude/longitude, НЕ добавляем lat/lon чтобы не было путаницы)
    latitude = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(-90.0), MaxValueValidator(90.0)],
    )
    longitude = models.FloatField(
        null=True,
        blank=True,
        validators=[MinValueValidator(-180.0), MaxValueValidator(180.0)],
    )

    timezone = models.CharField(max_length=100, default="Asia/Almaty")

    # === Паспорт станции (MVP) ===
    # Мощности (кВт)
    capacity_dc_kw = models.FloatField(default=1000.0, validators=[MinValueValidator(0.0)])
    capacity_ac_kw = models.FloatField(default=1000.0, validators=[MinValueValidator(0.0)])

    # PR
    pr_default = models.FloatField(
        default=0.88,
        validators=[MinValueValidator(0.10), MaxValueValidator(1.00)],
    )

    # Геометрия
    tilt_deg = models.FloatField(
        default=30.0,
        validators=[MinValueValidator(0.0), MaxValueValidator(90.0)],
    )
    azimuth_deg = models.FloatField(
        default=180.0,
        validators=[MinValueValidator(0.0), MaxValueValidator(360.0)],
    )

    # Потери (в сумме)
    losses_total_pct = models.FloatField(
        default=10.0,
        validators=[MinValueValidator(0.0), MaxValueValidator(40.0)],
    )

    history_source = models.ForeignKey(
        "self",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="history_dependents",
        help_text="Использовать историю другой станции при отсутствии своей.",
    )
    history_scale_by_capacity = models.BooleanField(
        default=True,
        help_text="Масштабировать историю по отношению мощностей станций.",
    )
    auto_history_enabled = models.BooleanField(
        default=False,
        help_text="Автоматически подтягивать историю из общей папки.",
    )
    auto_history_folder = models.CharField(
        max_length=500,
        default="/mnt/share",
        help_text="Путь к папке с D222*.csv.gz и FusionSolar .xlsx отчетами.",
    )
    sort_order = models.PositiveIntegerField(default=0, db_index=True)

    def save(self, *args, **kwargs):
        if self.pk is None and self.sort_order == 0:
            last_order = (
                Station.objects.filter(org=self.org)
                .aggregate(models.Max("sort_order"))
                .get("sort_order__max")
                or 0
            )
            self.sort_order = last_order + 1
        super().save(*args, **kwargs)

    def clean(self):
        super().clean()
        if (
            self.capacity_ac_kw is not None
            and self.capacity_dc_kw is not None
            and self.capacity_ac_kw > self.capacity_dc_kw
        ):
            raise ValidationError({"capacity_ac_kw": "AC мощность не должна быть больше DC мощности."})

    def __str__(self):
        return f"{self.name} ({self.capacity_mw} MW)"
