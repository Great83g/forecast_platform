from __future__ import annotations

from datetime import datetime, time, timedelta

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.utils import timezone

from dashboard.services.history_autofill import upsert_station_history_from_share
from solar.models import SolarRecord
from stations.models import Station


class Command(BaseCommand):
    help = (
        "Force re-import station auto-history from its configured share folder. "
        "Useful after deploying parser fixes, for example SES 1.2 MW hour alignment."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "station_ids",
            nargs="*",
            type=int,
            help="Station ids to re-import. Omit with --all-auto-history to process all enabled stations.",
        )
        parser.add_argument(
            "--all-auto-history",
            action="store_true",
            help="Re-import all stations with auto_history_enabled=True.",
        )
        parser.add_argument(
            "--from-date",
            dest="from_date",
            help="Optional first local date to clear before re-import, YYYY-MM-DD.",
        )
        parser.add_argument(
            "--to-date",
            dest="to_date",
            help="Optional last local date to clear before re-import, YYYY-MM-DD. Defaults to --from-date.",
        )
        parser.add_argument(
            "--clear-window",
            action="store_true",
            help=(
                "Delete existing main-history records for the selected date window before re-import. "
                "Use this after a time-shift bug so stale shifted rows are removed."
            ),
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Print what would be cleared/imported without changing the database.",
        )

    def handle(self, *args, **options):
        station_ids = options["station_ids"]
        use_all = options["all_auto_history"]
        if station_ids and use_all:
            raise CommandError("Use either station_ids or --all-auto-history, not both.")
        if not station_ids and not use_all:
            raise CommandError("Pass at least one station_id or use --all-auto-history.")

        clear_window = bool(options["clear_window"])
        dry_run = bool(options["dry_run"])
        from_date, to_date = self._parse_date_window(options["from_date"], options["to_date"])
        if clear_window and from_date is None:
            raise CommandError("--clear-window requires --from-date YYYY-MM-DD.")

        stations = self._get_stations(station_ids, use_all)
        total_rows = 0
        total_deleted = 0

        for station in stations:
            with transaction.atomic():
                deleted = 0
                if clear_window:
                    qs = self._history_window_qs(station, from_date, to_date)
                    deleted = qs.count()
                    if not dry_run:
                        qs.delete()

                if dry_run:
                    rows = 0
                    self.stdout.write(
                        self.style.WARNING(
                            f"DRY RUN station_id={station.pk} name={station.name!r}: "
                            f"would_delete={deleted}, would_reimport_from={station.auto_history_folder!r}"
                        )
                    )
                else:
                    rows = int(upsert_station_history_from_share(station) or 0)
                    self.stdout.write(
                        self.style.SUCCESS(
                            f"station_id={station.pk} name={station.name!r}: deleted={deleted}, upserted={rows}"
                        )
                    )

                total_deleted += deleted
                total_rows += rows

        self.stdout.write(f"done: stations={len(stations)}, deleted={total_deleted}, upserted={total_rows}")

    def _get_stations(self, station_ids: list[int], use_all: bool):
        if use_all:
            return list(Station.objects.filter(auto_history_enabled=True).order_by("pk"))

        stations = list(Station.objects.filter(pk__in=station_ids).order_by("pk"))
        found_ids = {station.pk for station in stations}
        missing_ids = sorted(set(station_ids) - found_ids)
        if missing_ids:
            raise CommandError(f"Station(s) not found: {missing_ids}")
        return stations

    def _parse_date_window(self, from_value: str | None, to_value: str | None):
        if not from_value and not to_value:
            return None, None
        if not from_value:
            raise CommandError("--to-date requires --from-date.")

        try:
            from_date = datetime.strptime(from_value, "%Y-%m-%d").date()
            to_date = datetime.strptime(to_value or from_value, "%Y-%m-%d").date()
        except ValueError as exc:
            raise CommandError("Date format must be YYYY-MM-DD.") from exc

        if to_date < from_date:
            raise CommandError("--to-date must be greater than or equal to --from-date.")
        return from_date, to_date

    def _history_window_qs(self, station: Station, from_date, to_date):
        start_naive = datetime.combine(from_date, time.min)
        end_exclusive_naive = datetime.combine(to_date + timedelta(days=1), time.min)
        current_tz = timezone.get_current_timezone()
        start = timezone.make_aware(start_naive, current_tz)
        end_exclusive = timezone.make_aware(end_exclusive_naive, current_tz)
        return SolarRecord.objects.filter(
            station=station,
            history_scope=SolarRecord.HISTORY_SCOPE_MAIN,
            timestamp__gte=start,
            timestamp__lt=end_exclusive,
        )
