from datetime import time

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("stations", "0011_station_auto_history_script"),
    ]

    operations = [
        migrations.AddField(
            model_name="station",
            name="auto_history_last_run_date",
            field=models.DateField(
                blank=True,
                editable=False,
                help_text="Служебное поле: дата последней автопроверки истории.",
                null=True,
            ),
        ),
        migrations.AddField(
            model_name="station",
            name="auto_history_run_time",
            field=models.TimeField(
                default=time(6, 0),
                help_text="Ежедневное время запуска автообновления истории для станции.",
            ),
        ),
    ]
