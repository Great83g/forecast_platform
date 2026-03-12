from __future__ import annotations

from django.core.management.base import BaseCommand, CommandError

from stations.models import Station
from dashboard.services.train_models import train_models_for_station


class Command(BaseCommand):
    help = "Train XGB + NeuralProphet models for a station in a standalone process."

    def add_arguments(self, parser):
        parser.add_argument("station_id", type=int, help="Station ID")

    def handle(self, *args, **options):
        station_id = options["station_id"]
        station = Station.objects.filter(pk=station_id).first()
        if station is None:
            raise CommandError(f"Station with id={station_id} not found")

        self.stdout.write(self.style.NOTICE(f"[TRAIN] start station={station_id}"))
        n_rows, np_path, xgb_path = train_models_for_station(station)
        self.stdout.write(
            self.style.SUCCESS(
                f"[TRAIN] done station={station_id}; rows={n_rows}; np={np_path}; xgb={xgb_path}"
            )
        )
