from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("dashboard", "0004_forecastschedule_manual_snow"),
    ]

    operations = [
        migrations.AddField(
            model_name="forecastschedule",
            name="horizon_mode",
            field=models.CharField(default="legacy", max_length=32),
        ),
    ]
