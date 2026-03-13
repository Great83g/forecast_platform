from __future__ import annotations

import fcntl
import logging
import os
import sys
import threading
import time
from pathlib import Path

from django.conf import settings

from dashboard.services.forecast_scheduler import run_scheduled_forecasts
from dashboard.services.history_autofill import run_auto_history_updates


logger = logging.getLogger(__name__)

_BACKGROUND_THREAD: threading.Thread | None = None
_BACKGROUND_THREAD_LOCK = threading.Lock()


def _is_runserver_reloader_parent() -> bool:
    return len(sys.argv) > 1 and sys.argv[1] == "runserver" and os.environ.get("RUN_MAIN") != "true"


def _background_scheduler_enabled() -> bool:
    if "test" in sys.argv:
        return False
    if _is_runserver_reloader_parent():
        return False
    return bool(getattr(settings, "FORECAST_BACKGROUND_SCHEDULER_ENABLED", True))


def _run_tick_with_file_lock() -> bool:
    lock_path = Path(getattr(settings, "BASE_DIR", Path.cwd())) / ".forecast_scheduler.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with lock_path.open("w") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False

        updated_rows = int(run_auto_history_updates() or 0)
        forecast_count = int(run_scheduled_forecasts() or 0)
        logger.info(
            "Background scheduler tick completed auto_history_rows=%s forecast_count=%s",
            updated_rows,
            forecast_count,
        )
        return True


def _scheduler_loop(interval_seconds: int) -> None:
    logger.info("Background scheduler started interval_seconds=%s", interval_seconds)

    while True:
        started_at = time.monotonic()
        try:
            _run_tick_with_file_lock()
        except Exception:
            logger.exception("Background scheduler tick failed")

        elapsed = time.monotonic() - started_at
        sleep_for = max(1, interval_seconds - int(elapsed))
        time.sleep(sleep_for)


def start_background_scheduler() -> None:
    if not _background_scheduler_enabled():
        return

    global _BACKGROUND_THREAD
    with _BACKGROUND_THREAD_LOCK:
        if _BACKGROUND_THREAD and _BACKGROUND_THREAD.is_alive():
            return

        interval_seconds = int(getattr(settings, "FORECAST_BACKGROUND_SCHEDULER_INTERVAL_SECONDS", 60))
        interval_seconds = max(15, interval_seconds)

        _BACKGROUND_THREAD = threading.Thread(
            target=_scheduler_loop,
            args=(interval_seconds,),
            name="forecast-background-scheduler",
            daemon=True,
        )
        _BACKGROUND_THREAD.start()
