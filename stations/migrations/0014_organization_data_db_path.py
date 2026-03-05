from pathlib import Path
import sqlite3

from django.conf import settings
from django.db import migrations, models


ORG_DB_DIRNAME = "org_databases"


def ensure_org_sqlite(org_id: int) -> str:
    db_path = Path(settings.BASE_DIR) / ORG_DB_DIRNAME / f"org_{org_id}.sqlite3"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS org_metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO org_metadata(key, value)
            VALUES ('organization_id', ?)
            """,
            (str(org_id),),
        )
        conn.commit()

    return str(db_path)


def fill_paths(apps, schema_editor):
    Organization = apps.get_model("stations", "Organization")
    for org in Organization.objects.filter(data_db_path="").iterator():
        org.data_db_path = ensure_org_sqlite(org.id)
        org.save(update_fields=["data_db_path"])


def clear_paths(apps, schema_editor):
    Organization = apps.get_model("stations", "Organization")
    Organization.objects.exclude(data_db_path="").update(data_db_path="")


class Migration(migrations.Migration):

    dependencies = [
        ("stations", "0013_station_auto_history_monitor_fields"),
    ]

    operations = [
        migrations.AddField(
            model_name="organization",
            name="data_db_path",
            field=models.CharField(
                blank=True,
                default="",
                help_text="Путь к выделенной SQLite базе организации.",
                max_length=500,
            ),
        ),
        migrations.RunPython(fill_paths, clear_paths),
    ]
