from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("stations", "0015_station_data_shift_hours"),
    ]

    operations = [
        migrations.AddField(
            model_name="station",
            name="station_kind",
            field=models.CharField(
                choices=[("solar", "Солнечная"), ("wind", "Ветровая")],
                db_index=True,
                default="solar",
                max_length=16,
            ),
        ),
    ]
