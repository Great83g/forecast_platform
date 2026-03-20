from __future__ import annotations

from pathlib import Path

from django.utils.text import slugify


def canonical_station_model_dir(model_dir: Path, station) -> Path:
    return Path(model_dir) / str(getattr(station, "pk"))


def legacy_station_model_dir(model_dir: Path, station) -> Path:
    slug = slugify(getattr(station, "name", "")) or "station"
    return Path(model_dir) / f"{getattr(station, 'pk')}_{slug}"


def resolve_station_model_dir(model_dir: Path, station, *, create: bool = False) -> Path:
    base = Path(model_dir)
    canonical = canonical_station_model_dir(base, station)
    legacy = legacy_station_model_dir(base, station)

    if canonical.exists():
        return canonical
    if legacy.exists():
        return legacy
    if create:
        canonical.mkdir(parents=True, exist_ok=True)
        return canonical
    return canonical
