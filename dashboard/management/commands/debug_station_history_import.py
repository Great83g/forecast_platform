from __future__ import annotations

from pathlib import Path

from django.core.management.base import BaseCommand, CommandError

from dashboard.services.history_autofill import (
    _load_station_history_builder,
    _resolve_station_share_folder,
    collect_share_history_dataframe,
)
from stations.models import Station


class Command(BaseCommand):
    help = "Debug auto-history import for a station and print parser output stats."

    def add_arguments(self, parser):
        parser.add_argument("station_id", type=int)
        parser.add_argument("--head", type=int, default=10)

    def handle(self, *args, **options):
        station_id = options["station_id"]
        head_n = options["head"]

        try:
            station = Station.objects.get(pk=station_id)
        except Station.DoesNotExist as exc:
            raise CommandError(f"Station not found: {station_id}") from exc

        folder = _resolve_station_share_folder(station)
        builder = _load_station_history_builder(station)

        self.stdout.write(f"station_id={station.pk} name={station.name}")
        self.stdout.write(f"auto_history_enabled={station.auto_history_enabled}")
        self.stdout.write(f"auto_history_folder(raw)={station.auto_history_folder}")
        self.stdout.write(f"auto_history_folder(resolved)={folder}")
        self.stdout.write(f"auto_history_script={station.auto_history_script!r}")
        self.stdout.write(f"builder={builder}")

        if not folder.exists():
            raise CommandError(f"Resolved folder does not exist: {folder}")

        files = [
            p
            for p in sorted(folder.rglob("*"))
            if p.is_file() and p.suffix.lower() in {".xlsx", ".xlsm", ".xltx", ".xltm", ".csv", ".gz"}
        ]
        self.stdout.write(f"candidate_files={len(files)}")
        for p in files[:20]:
            rel = p.relative_to(folder) if folder in p.parents or p == folder else Path(p.name)
            self.stdout.write(f" - {rel}")

        if builder is not None:
            df = builder(station)
        else:
            df = collect_share_history_dataframe(folder)

        self.stdout.write(f"rows={len(df)}")
        if df.empty:
            self.stdout.write(self.style.WARNING("DataFrame is empty."))
            return

        if "ds" in df.columns:
            self.stdout.write(f"ds_min={df['ds'].min()} ds_max={df['ds'].max()}")

        self.stdout.write("columns=" + ", ".join(map(str, df.columns)))
        self.stdout.write(df.head(head_n).to_string(index=False))
