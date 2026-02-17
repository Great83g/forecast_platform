from django.db import migrations, models


def fill_sort_order(apps, schema_editor):
    Station = apps.get_model("stations", "Station")
    Organization = apps.get_model("stations", "Organization")

    for org in Organization.objects.all().iterator():
        station_ids = list(
            Station.objects.filter(org=org)
            .order_by("id")
            .values_list("id", flat=True)
        )
        for idx, station_id in enumerate(station_ids, start=1):
            Station.objects.filter(id=station_id).update(sort_order=idx)


class Migration(migrations.Migration):

    dependencies = [
        ("stations", "0008_organization_subscription_status"),
    ]

    operations = [
        migrations.AddField(
            model_name="station",
            name="sort_order",
            field=models.PositiveIntegerField(db_index=True, default=0),
        ),
        migrations.RunPython(fill_sort_order, migrations.RunPython.noop),
    ]
