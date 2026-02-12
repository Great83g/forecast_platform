from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ("stations", "0005_organizationmember_org_trial"),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name="OrganizationInvitation",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("invited_email", models.EmailField(max_length=254)),
                (
                    "role",
                    models.CharField(
                        choices=[
                            ("owner", "Owner"),
                            ("admin", "Admin"),
                            ("analyst", "Analyst"),
                            ("viewer", "Viewer"),
                        ],
                        default="viewer",
                        max_length=16,
                    ),
                ),
                ("token", models.CharField(db_index=True, max_length=64, unique=True)),
                (
                    "status",
                    models.CharField(
                        choices=[("pending", "Pending"), ("accepted", "Accepted"), ("cancelled", "Cancelled")],
                        default="pending",
                        max_length=16,
                    ),
                ),
                ("expires_at", models.DateTimeField()),
                ("accepted_at", models.DateTimeField(blank=True, null=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                (
                    "accepted_by",
                    models.ForeignKey(
                        blank=True,
                        null=True,
                        on_delete=django.db.models.deletion.SET_NULL,
                        related_name="accepted_org_invitations",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
                (
                    "invited_by",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="sent_org_invitations",
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
                (
                    "organization",
                    models.ForeignKey(
                        on_delete=django.db.models.deletion.CASCADE,
                        related_name="invitations",
                        to="stations.organization",
                    ),
                ),
            ],
        ),
        migrations.AddIndex(
            model_name="organizationinvitation",
            index=models.Index(fields=["organization", "status"], name="stations_or_organiz_14b6a7_idx"),
        ),
        migrations.AddIndex(
            model_name="organizationinvitation",
            index=models.Index(fields=["invited_email", "status"], name="stations_or_invited_8e7f65_idx"),
        ),
    ]
