from __future__ import annotations

import re
import shutil
from pathlib import Path

from django.utils.text import slugify


MODEL_ARTIFACT_FILENAMES = (
    "np_model.np",
    "np_model.meta.json",
    "xgb_model.json",
    "xgb_model.meta.json",
)
LEGACY_ROOT_MODEL_PATTERNS = {
    "legacy_np": "np_model_{station_id}.np",
    "legacy_np_meta": "np_model_{station_id}.meta.json",
    "legacy_xgb": "xgb_model_{station_id}.json",
    "legacy_xgb_meta": "xgb_model_{station_id}.meta.json",
}
LEGACY_DIR_RE = re.compile(r"^(?P<station_id>\d+)_")


def canonical_station_model_dir(model_dir: Path, station) -> Path:
    return Path(model_dir) / str(getattr(station, "pk"))


def legacy_station_model_dir(model_dir: Path, station) -> Path:
    slug = slugify(getattr(station, "name", "")) or "station"
    return Path(model_dir) / f"{getattr(station, 'pk')}_{slug}"


def legacy_root_model_paths(model_dir: Path, station) -> dict[str, Path]:
    base = Path(model_dir)
    station_id = getattr(station, "pk")
    return {
        key: base / pattern.format(station_id=station_id)
        for key, pattern in LEGACY_ROOT_MODEL_PATTERNS.items()
    }


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
            for name in MODEL_ARTIFACT_FILENAMES
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


def _move_if_needed(src: Path, dst: Path, moved: list[str], removed: list[str]) -> None:
    if not src.exists():
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if src.is_dir():
            shutil.rmtree(src)
        else:
            src.unlink()
        removed.append(str(src))
        return

    shutil.move(str(src), str(dst))
    moved.append(f"{src} -> {dst}")


def normalize_station_model_artifacts(model_dir: Path, station) -> dict[str, list[str]]:
    base = Path(model_dir)
    canonical = canonical_station_model_dir(base, station)
    canonical.mkdir(parents=True, exist_ok=True)

    moved: list[str] = []
    removed: list[str] = []

    legacy_dirs = sorted(
        path for path in base.glob(f"{getattr(station, 'pk')}_*") if path.is_dir()
    )
    for legacy_dir in legacy_dirs:
        for filename in MODEL_ARTIFACT_FILENAMES:
            _move_if_needed(legacy_dir / filename, canonical / filename, moved, removed)
        shutil.rmtree(legacy_dir, ignore_errors=True)
        removed.append(str(legacy_dir))

    root_legacy = legacy_root_model_paths(base, station)
    destination_map = {
        "legacy_np": canonical / "np_model.np",
        "legacy_np_meta": canonical / "np_model.meta.json",
        "legacy_xgb": canonical / "xgb_model.json",
        "legacy_xgb_meta": canonical / "xgb_model.meta.json",
    }
    for key, src in root_legacy.items():
        _move_if_needed(src, destination_map[key], moved, removed)

    return {"moved": moved, "removed": removed, "canonical_dir": [str(canonical)]}


def cleanup_orphan_model_artifacts(model_dir: Path, stations) -> dict[str, list[str]]:
    base = Path(model_dir)
    base.mkdir(parents=True, exist_ok=True)
    station_ids = {str(getattr(station, "pk")) for station in stations}
    removed: list[str] = []

    for child in base.iterdir():
        if child.name.startswith("train_station_") and child.suffix == ".log":
            continue

        if child.is_dir() and child.name.isdigit() and child.name not in station_ids:
            shutil.rmtree(child, ignore_errors=True)
            removed.append(str(child))
            continue

        match = LEGACY_DIR_RE.match(child.name)
        if child.is_dir() and match and match.group("station_id") not in station_ids:
            shutil.rmtree(child, ignore_errors=True)
            removed.append(str(child))
            continue

        for pattern in LEGACY_ROOT_MODEL_PATTERNS.values():
            prefix, suffix = pattern.split("{station_id}")
            if child.is_file() and child.name.startswith(prefix) and child.name.endswith(suffix):
                station_id = child.name[len(prefix): len(child.name) - len(suffix) if suffix else None]
                if station_id not in station_ids:
                    child.unlink(missing_ok=True)
                    removed.append(str(child))
                break

    return {"removed": removed}


def normalize_model_cache(model_dir: Path, stations) -> dict[str, list[str]]:
    base = Path(model_dir)
    base.mkdir(parents=True, exist_ok=True)
    moved: list[str] = []
    removed: list[str] = []

    station_list = list(stations)
    for station in station_list:
        result = normalize_station_model_artifacts(base, station)
        moved.extend(result["moved"])
        removed.extend(result["removed"])

    orphan_result = cleanup_orphan_model_artifacts(base, station_list)
    removed.extend(orphan_result["removed"])

    return {"moved": moved, "removed": removed}
