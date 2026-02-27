import logging
import os
import secrets
import stat
import tempfile
from datetime import time
from pathlib import Path

from django.db import models
from django.contrib.auth.models import User
import re

from django.core.validators import MinValueValidator, MaxValueValidator
from django.core.exceptions import ValidationError
from django.utils import timezone


logger = logging.getLogger(__name__)


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
    auto_history_script = models.CharField(
        max_length=255,
        blank=True,
        default="",
        help_text=(
            "Индивидуальный скрипт для автоистории. Формат: "
            "python.module:function_name (оставьте пустым для стандартного обработчика)."
        ),
    )
    auto_history_run_time = models.TimeField(
        default=time(6, 0),
        help_text="Ежедневное время запуска автообновления истории для станции.",
    )
    auto_history_last_run_date = models.DateField(
        null=True,
        blank=True,
        editable=False,
        help_text="Служебное поле: дата последней автопроверки истории.",
    )
    auto_history_last_check_at = models.DateTimeField(
        null=True,
        blank=True,
        editable=False,
        help_text="Служебное поле: дата/время последнего тика автоистории.",
    )
    auto_history_last_status = models.CharField(
        max_length=32,
        blank=True,
        default="",
        help_text="Служебное поле: статус последнего тика автоистории.",
    )
    auto_history_last_rows = models.IntegerField(
        default=0,
        help_text="Служебное поле: сколько строк обновлено на последнем тике.",
    )
    auto_history_last_message = models.TextField(
        blank=True,
        default="",
        help_text="Служебное поле: сообщение последнего тика автоистории.",
    )
    sort_order = models.PositiveIntegerField(default=0, db_index=True)

    @staticmethod
    def _build_auto_history_folder(station_name: str, org_id: int | None = None) -> str:
        base_folder = "/mnt/share"
        normalized_name = re.sub(r"[\\/]+", "_", (station_name or "").strip())
        normalized_name = re.sub(r"\s+", "_", normalized_name).strip("._")
        path_parts = []
        if org_id:
            path_parts.append(f"org_{org_id}")
        if normalized_name:
            path_parts.append(normalized_name)
        if not path_parts:
            return base_folder
        return f"{base_folder}/{'/'.join(path_parts)}"

    @staticmethod
    def _build_fallback_auto_history_folder(station_name: str, org_id: int | None = None) -> str:
        base_folder = Path(tempfile.gettempdir()) / "forecast_platform_auto_history"
        normalized_name = re.sub(r"[\\/]+", "_", (station_name or "").strip())
        normalized_name = re.sub(r"\s+", "_", normalized_name).strip("._")
        path_parts = []
        if org_id:
            path_parts.append(f"org_{org_id}")
        if normalized_name:
            path_parts.append(normalized_name)
        if not path_parts:
            return str(base_folder)
        return str(base_folder / Path(*path_parts))

    def _build_preferred_auto_history_folder(self) -> str:
        fallback_root = str(Path(tempfile.gettempdir()) / "forecast_platform_auto_history")
        fallback_prefix = f"{fallback_root.rstrip('/')}/"
        has_org_fallback = False
        if self.org_id:
            has_org_fallback = type(self).objects.filter(
                org_id=self.org_id,
                auto_history_folder__startswith=fallback_prefix,
            ).exclude(pk=self.pk).exists()
        if has_org_fallback:
            return self._build_fallback_auto_history_folder(self.name, self.org_id)
        return self._build_auto_history_folder(self.name, self.org_id)

    def ensure_import_folder(self) -> bool:
        folder = (self.auto_history_folder or "").strip()
        if not folder:
            self._last_import_folder_error = "Папка автоимпорта не задана."
            return False

        if folder.rstrip("/") == "/mnt/share":
            folder = self._build_preferred_auto_history_folder()
            self.auto_history_folder = folder
            if self.pk:
                type(self).objects.filter(pk=self.pk).update(auto_history_folder=folder)

        target = Path(folder)
        parent = target.parent

        try:
            target.mkdir(parents=True, exist_ok=True)
            if target.exists():
                self._last_import_folder_error = ""
                return True

            self._last_import_folder_error = f"Папка не появилась после создания: {folder}"
            return False
        except OSError as exc:
            share_root = Path("/mnt/share")
            is_share_permission_error = isinstance(exc, PermissionError) and (target == share_root or share_root in target.parents)
            if is_share_permission_error:
                fallback_folder = self._build_fallback_auto_history_folder(self.name, self.org_id)
                fallback_target = Path(fallback_folder)
                try:
                    fallback_target.mkdir(parents=True, exist_ok=True)
                    self.auto_history_folder = fallback_folder
                    if self.pk:
                        type(self).objects.filter(pk=self.pk).update(auto_history_folder=fallback_folder)
                    self._last_import_folder_error = (
                        f"PermissionError on /mnt/share ({exc}); switched to fallback folder: {fallback_folder}"
                    )
                    logger.warning(
                        "Switching station auto-history folder to fallback station_id=%s from=%s to=%s",
                        self.pk,
                        folder,
                        fallback_folder,
                    )
                    return True
                except OSError:
                    logger.exception(
                        "Failed to create fallback auto-history folder station_id=%s fallback=%s",
                        self.pk,
                        fallback_folder,
                    )

            parent_exists = parent.exists()
            parent_writable = os.access(parent, os.W_OK | os.X_OK) if parent_exists else False
            process_uid = os.geteuid()
            process_gid = os.getegid()
            parent_owner = "?"
            parent_group = "?"
            parent_mode = "?"
            nearest_existing_parent = parent
            if parent_exists:
                parent_stat = parent.stat()
                parent_owner = str(parent_stat.st_uid)
                parent_group = str(parent_stat.st_gid)
                parent_mode = stat.filemode(parent_stat.st_mode)
            else:
                for candidate in parent.parents:
                    if candidate.exists():
                        nearest_existing_parent = candidate
                        parent_stat = candidate.stat()
                        parent_owner = str(parent_stat.st_uid)
                        parent_group = str(parent_stat.st_gid)
                        parent_mode = stat.filemode(parent_stat.st_mode)
                        parent_writable = os.access(candidate, os.W_OK | os.X_OK)
                        break

            hint = ""
            if is_share_permission_error:
                hint = (
                    " hint=Недостаточно прав для записи в /mnt/share. "
                    "Проверьте доступ к шаре или используйте fallback-путь в /tmp."
                )
            self._last_import_folder_error = (
                f"{exc.__class__.__name__}: {exc}. "
                f"parent={parent} exists={parent_exists} writable={parent_writable} "
                f"nearest_existing_parent={nearest_existing_parent} "
                f"process_uid={process_uid} process_gid={process_gid} "
                f"parent_uid={parent_owner} parent_gid={parent_group} parent_mode={parent_mode}"
                f"{hint}"
            )
            logger.warning(
                "Cannot create auto-history folder station_id=%s folder=%s error=%s",
                self.pk,
                folder,
                self._last_import_folder_error,
                exc_info=True,
            )
            return False

    def save(self, *args, **kwargs):
        if self.pk:
            previous = (
                Station.objects.filter(pk=self.pk)
                .values(
                    "auto_history_enabled",
                    "auto_history_folder",
                    "auto_history_script",
                    "auto_history_run_time",
                    "history_source_id",
                    "history_scale_by_capacity",
                )
                .first()
            )
            if previous:
                auto_history_config_changed = any(
                    [
                        previous["auto_history_enabled"] != self.auto_history_enabled,
                        (previous["auto_history_folder"] or "") != (self.auto_history_folder or ""),
                        (previous["auto_history_script"] or "") != (self.auto_history_script or ""),
                        previous["auto_history_run_time"] != self.auto_history_run_time,
                        previous["history_source_id"] != self.history_source_id,
                        previous["history_scale_by_capacity"] != self.history_scale_by_capacity,
                    ]
                )
                if auto_history_config_changed:
                    self.auto_history_last_run_date = None

        if (self.auto_history_folder or "").rstrip("/") == "/mnt/share":
            self.auto_history_folder = self._build_preferred_auto_history_folder()
        if self.pk is None and self.sort_order == 0:
            last_order = (
                Station.objects.filter(org=self.org)
                .aggregate(models.Max("sort_order"))
                .get("sort_order__max")
                or 0
            )
            self.sort_order = last_order + 1
        super().save(*args, **kwargs)
        self.ensure_import_folder()

    @classmethod
    def ensure_all_import_folders(cls, station_ids: list[int] | None = None) -> int:
        qs = cls.objects.all().only("id", "name", "org_id", "auto_history_folder")
        if station_ids:
            qs = qs.filter(id__in=station_ids)

        count = 0
        for station in qs.iterator():
            if station.ensure_import_folder():
                count += 1
        return count


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
