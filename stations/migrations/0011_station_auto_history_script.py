from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("stations", "0010_station_auto_history_fields"),
    ]

    operations = [
        migrations.AddField(
            model_name="station",
            name="auto_history_script",
            field=models.CharField(
                blank=True,
                default="",
                help_text=(
                    "Индивидуальный скрипт для автоистории. Формат: "
                    "python.module:function_name (оставьте пустым для стандартного обработчика)."
                ),
                max_length=255,
            ),
        ),
    ]
