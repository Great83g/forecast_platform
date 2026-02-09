from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ("stations", "0003_rename_ses_station"),
    ]

    operations = [
        migrations.AddField(
            model_name="station",
            name="history_source",
            field=models.ForeignKey(
                blank=True,
                help_text="Использовать историю другой станции при отсутствии своей.",
                null=True,
                on_delete=django.db.models.deletion.SET_NULL,
                related_name="history_dependents",
                to="stations.station",
            ),
        ),
        migrations.AddField(
            model_name="station",
            name="history_scale_by_capacity",
            field=models.BooleanField(
                default=True,
                help_text="Масштабировать историю по отношению мощностей станций.",
            ),
        ),
    ]
