from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("stations", "0009_station_sort_order"),
    ]

    operations = [
        migrations.AddField(
            model_name="station",
            name="auto_history_enabled",
            field=models.BooleanField(
                default=False,
                help_text="Автоматически подтягивать историю из общей папки.",
            ),
        ),
        migrations.AddField(
            model_name="station",
            name="auto_history_folder",
            field=models.CharField(
                default="/mnt/share",
                help_text="Путь к папке с D222*.csv.gz и FusionSolar .xlsx отчетами.",
                max_length=500,
            ),
        ),
    ]
