from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("stations", "0014_organization_data_db_path"),
    ]

    operations = [
        migrations.AddField(
            model_name="station",
            name="data_shift_hours",
            field=models.IntegerField(
                default=0,
                help_text="Сдвиг данных станции в часах (прогноз/автоистория/сопоставление с фактом).",
                validators=[MinValueValidator(-12), MaxValueValidator(12)],
            ),
        ),
    ]
