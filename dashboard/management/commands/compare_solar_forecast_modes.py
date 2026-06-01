from __future__ import annotations

from datetime import datetime

from django.core.management.base import BaseCommand, CommandError

from dashboard.services.forecast_diagnostics import compare_forecast_modes, comparison_rows_to_dataframe
from solar.models import SolarForecast
from stations.models import Station


class Command(BaseCommand):
    help = "Compare operational and postfact solar forecasts hour by hour."

    def add_arguments(self, parser):
        station_group = parser.add_mutually_exclusive_group(required=True)
        station_group.add_argument("--station-id", type=int)
        station_group.add_argument("--station", dest="station_name")
        parser.add_argument("--from", dest="date_from", required=True, help="Start date, YYYY-MM-DD")
        parser.add_argument("--to", dest="date_to", required=True, help="End date, YYYY-MM-DD")
        parser.add_argument("--forecast-scope", default=SolarForecast.SCOPE_MAIN)
        parser.add_argument("--postfact-scope", default=SolarForecast.SCOPE_TEST)
        parser.add_argument("--threshold", type=float, default=5.0, help="Log relative differences above this percent")

    def handle(self, *args, **options):
        try:
            date_from = datetime.strptime(options["date_from"], "%Y-%m-%d").date()
            date_to = datetime.strptime(options["date_to"], "%Y-%m-%d").date()
        except ValueError as exc:
            raise CommandError("Dates must be in YYYY-MM-DD format") from exc
        if date_to < date_from:
            raise CommandError("--to must be greater than or equal to --from")

        if options.get("station_id"):
            station = Station.objects.filter(pk=options["station_id"]).first()
        else:
            station = Station.objects.filter(name=options["station_name"]).first()
        if station is None:
            raise CommandError("Station not found")

        rows = compare_forecast_modes(
            station,
            date_from,
            date_to,
            forecast_scope=options["forecast_scope"],
            postfact_scope=options["postfact_scope"],
            threshold_percent=options["threshold"],
        )
        df = comparison_rows_to_dataframe(rows)
        if df.empty:
            self.stdout.write("No forecast rows found for the requested range/scopes.")
            return

        self.stdout.write(
            f"station={station.name} mount_type={station.mount_type} "
            f"AC_kW={station.capacity_ac_kw} DC_kW={station.capacity_dc_kw} "
            f"forecast_scope={options['forecast_scope']} postfact_scope={options['postfact_scope']}"
        )
        self.stdout.write(df.to_csv(index=False))
