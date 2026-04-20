from __future__ import annotations

from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

from dashboard.services.model_storage import normalize_model_cache
from stations.models import Station


class Command(BaseCommand):
    help = "Normalize models_cache: migrate legacy station artifacts into canonical dirs and delete orphaned leftovers."

    def handle(self, *args, **options):
        model_dir = Path(getattr(settings, "MODEL_DIR", Path(settings.BASE_DIR) / "models_cache"))
        stations = list(Station.objects.only("pk", "name").order_by("pk"))
        result = normalize_model_cache(model_dir, stations)

        self.stdout.write(f"MODEL_DIR={model_dir}")
        self.stdout.write(f"stations={len(stations)}")
        self.stdout.write(self.style.SUCCESS(f"moved={len(result['moved'])} removed={len(result['removed'])}"))

        for entry in result["moved"]:
            self.stdout.write(f"MOVE {entry}")
        for entry in result["removed"]:
            self.stdout.write(f"REMOVE {entry}")
