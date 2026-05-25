from datetime import time
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("dashboard", "0007_forecastschedule_last_email_tracking"),
    ]

    operations = [
        migrations.AddField(
            model_name="forecastschedule",
            name="test_enabled",
            field=models.BooleanField(default=False),
        ),
        migrations.AddField(
            model_name="forecastschedule",
            name="test_run_time",
            field=models.TimeField(default=time(7, 0)),
        ),
        migrations.AddField(
            model_name="forecastschedule",
            name="test_providers",
            field=models.CharField(blank=True, max_length=128),
        ),
        migrations.AddField(
            model_name="forecastschedule",
            name="last_test_run_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
    ]
