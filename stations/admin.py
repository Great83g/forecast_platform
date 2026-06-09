from django.contrib import admin

from .models import Organization, OrganizationInvitation, OrganizationMember, Station


@admin.register(Organization)
class OrganizationAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "owner", "data_db_filename", "data_db_path", "trial_ends_at", "is_active")
    search_fields = ("name", "owner__username", "owner__email")

    @admin.display(description="DB file")
    def data_db_filename(self, obj: Organization) -> str:
        if not obj.data_db_path:
            return ""
        return obj.data_db_path.rsplit("/", 1)[-1]


@admin.register(OrganizationMember)
class OrganizationMemberAdmin(admin.ModelAdmin):
    list_display = ("id", "organization", "user", "role", "created_at")
    list_filter = ("role",)
    search_fields = ("organization__name", "user__username", "user__email")


@admin.register(OrganizationInvitation)
class OrganizationInvitationAdmin(admin.ModelAdmin):
    list_display = ("id", "organization", "invited_email", "role", "status", "expires_at", "created_at")
    list_filter = ("status", "role")
    search_fields = ("organization__name", "invited_email", "invited_by__username")


@admin.register(Station)
class StationAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "name",
        "org",
        "capacity_mw",
        "mount_type",
        "tracker_max_angle",
        "tracker_gcr",
        "tracker_backtrack",
        "tracker_albedo",
    )
    list_filter = ("mount_type", "tracker_backtrack")
    search_fields = ("name", "org__name")
