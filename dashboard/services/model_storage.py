from __future__ import annotations

from pathlib import Path

from django.utils.text import slugify


def canonical_station_model_dir(model_dir: Path, station) -> Path:
    return Path(model_dir) / str(getattr(station, "pk"))


def legacy_station_model_dir(model_dir: Path, station) -> Path:
    slug = slugify(getattr(station, "name", "")) or "station"
    return Path(model_dir) / f"{getattr(station, 'pk')}_{slug}"


def find_any_legacy_station_model_dir(model_dir: Path, station) -> Path | None:
    base = Path(model_dir)
    station_prefix = f"{getattr(station, 'pk')}_"
    candidates = [
        path
        for path in base.glob(f"{station_prefix}*")
        if path.is_dir()
    ]
    if not candidates:
        return None

    def _candidate_sort_key(path: Path) -> tuple[int, float, str]:
        model_files = sum(
            1
            for name in ("np_model.np", "xgb_model.json", "np_model.meta.json", "xgb_model.meta.json")
            if (path / name).exists()
        )
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        return (model_files, mtime, path.name)

    return max(candidates, key=_candidate_sort_key)


def describe_station_model_dir(model_dir: Path, station) -> tuple[Path, str]:
    base = Path(model_dir)
    canonical = canonical_station_model_dir(base, station)
    legacy = legacy_station_model_dir(base, station)

    if canonical.exists():
        return canonical, "canonical"
    if legacy.exists():
        return legacy, "legacy_current_slug"

    any_legacy = find_any_legacy_station_model_dir(base, station)
    if any_legacy is not None:
        if any_legacy == legacy:
            return any_legacy, "legacy_current_slug"
        return any_legacy, "legacy_previous_slug"

    return canonical, "canonical_missing"


def resolve_station_model_dir(model_dir: Path, station, *, create: bool = False) -> Path:
    canonical, source = describe_station_model_dir(model_dir, station)
    if source != "canonical_missing":
        return canonical
    if create:
        canonical.mkdir(parents=True, exist_ok=True)
        return canonical
    return canonical
