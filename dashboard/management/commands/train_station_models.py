from __future__ import annotations

from django.core.management.base import BaseCommand, CommandError

from stations.models import Station
from dashboard.services.train_models import train_models_for_station


class Command(BaseCommand):
    help = "Train XGB + NeuralProphet models for a station in a standalone process."

    def add_arguments(self, parser):
        parser.add_argument("station_id", nargs="?", type=int, help="Station ID")
        parser.add_argument(
            "--all",
            action="store_true",
            dest="train_all",
            help="Train models for all solar stations",
        )

    def handle(self, *args, **options):
        train_all = bool(options.get("train_all"))
        station_id = options.get("station_id")

        if train_all:
            stations = Station.objects.filter(station_kind=Station.KIND_SOLAR).order_by("id")
            if not stations.exists():
                raise CommandError("No solar stations found")
            for station in stations:
                self.stdout.write(self.style.NOTICE(f"[TRAIN] start station={station.pk}"))
                n_rows, np_path, xgb_path = train_models_for_station(station)
                self.stdout.write(
                    self.style.SUCCESS(
                        f"[TRAIN] done station={station.pk}; rows={n_rows}; np={np_path}; xgb={xgb_path}"
                    )
                )
            return

        if station_id is None:
            raise CommandError("station_id is required unless --all is provided")

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
