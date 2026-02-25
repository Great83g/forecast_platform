from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("stations", "0012_station_auto_history_schedule_fields"),
    ]

    operations = [
        migrations.AddField(
            model_name="station",
            name="auto_history_last_check_at",
            field=models.DateTimeField(blank=True, editable=False, help_text="Служебное поле: дата/время последнего тика автоистории.", null=True),
        ),
        migrations.AddField(
            model_name="station",
            name="auto_history_last_message",
            field=models.TextField(blank=True, default="", help_text="Служебное поле: сообщение последнего тика автоистории."),
        ),
        migrations.AddField(
            model_name="station",
            name="auto_history_last_rows",
            field=models.IntegerField(default=0, help_text="Служебное поле: сколько строк обновлено на последнем тике."),
        ),
        migrations.AddField(
            model_name="station",
            name="auto_history_last_status",
            field=models.CharField(blank=True, default="", help_text="Служебное поле: статус последнего тика автоистории.", max_length=32),
        ),
    ]
