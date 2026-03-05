from django.db.models.signals import post_save
from django.dispatch import receiver

from stations.models import Organization
from stations.org_database import ensure_org_sqlite


@receiver(post_save, sender=Organization)
def ensure_org_database_file(sender, instance: Organization, created: bool, **kwargs):
    if not created:
        return

    if instance.data_db_path:
        return

    db_path = ensure_org_sqlite(instance.id)
    Organization.objects.filter(pk=instance.pk, data_db_path="").update(data_db_path=db_path)
