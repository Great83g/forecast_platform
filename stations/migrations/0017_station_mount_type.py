# Generated manually for tracker mount-type support.

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("stations", "0016_station_station_kind"),
    ]

    operations = [
        migrations.AddField(
            model_name="station",
            name="mount_type",
            field=models.CharField(
                choices=[
                    ("fixed", "Фиксированный наклон"),
                    ("single_axis_tracker", "Одноосевой трекер"),
                    ("dual_axis_tracker", "Двухосевой трекер"),
                ],
                default="fixed",
                max_length=32,
            ),
        ),
    ]
