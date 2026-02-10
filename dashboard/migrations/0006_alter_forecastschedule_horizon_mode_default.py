from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("dashboard", "0005_forecastschedule_horizon_mode"),
    ]

    operations = [
        migrations.AlterField(
            model_name="forecastschedule",
            name="horizon_mode",
            field=models.CharField(default="weekday_calendar", max_length=32),
        ),
    ]
