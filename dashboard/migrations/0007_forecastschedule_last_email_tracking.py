from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("dashboard", "0006_alter_forecastschedule_horizon_mode_default"),
    ]

    operations = [
        migrations.AddField(
            model_name="forecastschedule",
            name="last_email_forecast_date",
            field=models.DateField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="forecastschedule",
            name="last_email_sent_at",
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="forecastschedule",
            name="last_email_status",
            field=models.TextField(blank=True),
        ),
    ]
