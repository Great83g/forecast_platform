import sqlite3
from pathlib import Path

from django.conf import settings


ORG_DB_DIRNAME = "org_databases"


def build_org_sqlite_path(org_id: int) -> Path:
    return Path(settings.BASE_DIR) / ORG_DB_DIRNAME / f"org_{org_id}.sqlite3"


def ensure_org_sqlite(org_id: int) -> str:
    db_path = build_org_sqlite_path(org_id)
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
