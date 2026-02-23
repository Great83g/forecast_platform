from django.core.management.base import BaseCommand

from stations.models import Station


class Command(BaseCommand):
    help = "Create missing auto-history import folders for stations."
    requires_system_checks = []

    def add_arguments(self, parser):
        parser.add_argument(
            "--station-id",
            action="append",
            dest="station_ids",
            type=int,
            help="Run only for specific station id (can be repeated).",
        )

    def handle(self, *args, **options):
        station_ids = options.get("station_ids") or None
        processed = Station.ensure_all_import_folders(station_ids=station_ids)
        self.stdout.write(self.style.SUCCESS(f"Import folders ensured (created/existing) for stations: {processed}"))
