#!/usr/bin/env bash
set -euo pipefail

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  cat <<'HELP'
Usage:
  bash deploy/apply_wind_forecast_runs_update.sh
  RUN_TESTS=0 bash deploy/apply_wind_forecast_runs_update.sh
  SKIP_SYNC=1 bash deploy/apply_wind_forecast_runs_update.sh

Environment variables:
  PROJECT_DIR               Project path (default: ~/forecast_platform)
  VENV_PATH                 Virtualenv activate path (default: $PROJECT_DIR/venv/bin/activate)
  RUN_TESTS=0               Skip the wind test suite after migrate/collectstatic
  SKIP_SYNC=1               Do not fetch/reset main; deploy the currently checked-out source
  BACKUP_DIR                Backup directory (default: $PROJECT_DIR/backups/wind_forecast_runs)
  GUNICORN_PID_FILE_DEFAULT Gunicorn pid file path (default: /run/gunicorn.pid)
  GUNICORN_PORT_DEFAULT     Fallback app port for restart (default: 8000)

Safety notes:
  - The script records current git status/diffs before syncing.
  - For the default SQLite database, it creates a timestamped backup before migrate.
  - Django migrations are idempotent: already-applied migrations are skipped by Django.
HELP
  exit 0
fi

PROJECT_DIR="${PROJECT_DIR:-$HOME/forecast_platform}"
VENV_PATH="${VENV_PATH:-$PROJECT_DIR/venv/bin/activate}"
GUNICORN_PID_FILE_DEFAULT="${GUNICORN_PID_FILE_DEFAULT:-/run/gunicorn.pid}"
GUNICORN_PORT_DEFAULT="${GUNICORN_PORT_DEFAULT:-8000}"
RUN_TESTS="${RUN_TESTS:-1}"
SKIP_SYNC="${SKIP_SYNC:-0}"
BACKUP_DIR="${BACKUP_DIR:-$PROJECT_DIR/backups/wind_forecast_runs}"
EXPECTED_MARKER="postfactum-forecast-panel"
STAMP="$(date +%Y%m%d_%H%M%S)"

log() {
  echo "[wind-forecast-runs] $*"
}

cd "$PROJECT_DIR"
mkdir -p "$BACKUP_DIR"

if [ -f "$VENV_PATH" ]; then
  # shellcheck disable=SC1090
  source "$VENV_PATH"
else
  echo "[wind-forecast-runs] ERROR: virtualenv activate script not found: $VENV_PATH" >&2
  exit 1
fi

log "Project: $PROJECT_DIR"
log "Backups: $BACKUP_DIR"

log "Saving pre-update git state"
git status --short > "$BACKUP_DIR/git_status_before_${STAMP}.txt" || true
git diff > "$BACKUP_DIR/git_diff_before_${STAMP}.patch" || true
git diff --cached > "$BACKUP_DIR/git_diff_cached_before_${STAMP}.patch" || true

if [ "$SKIP_SYNC" = "1" ]; then
  log "SKIP_SYNC=1, keeping current checkout"
else
  log "Fetching and forcing main to origin/main"
  git fetch --all --tags --prune
  git checkout main
  git reset --hard origin/main
fi

log "Current commit: $(git rev-parse --short HEAD) $(git log -1 --pretty=%s)"

if ! rg -q "$EXPECTED_MARKER" wind/templates/wind/station_forecast_list.html; then
  echo "[wind-forecast-runs] ERROR: expected marker '$EXPECTED_MARKER' is not in wind/templates/wind/station_forecast_list.html" >&2
  echo "[wind-forecast-runs] You are probably on an old origin/main. Merge/pull the PR commit first, then rerun this script." >&2
  exit 1
fi

log "Syntax checks"
python3 -m py_compile \
  manage.py \
  backend/urls.py \
  dashboard/services/open_meteo.py \
  dashboard/services/vc_weather.py \
  wind/models.py \
  wind/views.py \
  wind/services/forecast_runs.py \
  wind/services/forecasting.py \
  dashboard/services/forecast_scheduler.py \
  wind/tests.py
python3 -m compileall -q wind dashboard/services

log "Checking for missing migrations"
python3 manage.py makemigrations --check --dry-run

log "Backing up SQLite database when applicable"
DB_INFO="$(python3 - <<'PY'
import os
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.settings")
from django.conf import settings
cfg = settings.DATABASES["default"]
print(cfg.get("ENGINE", ""))
print(cfg.get("NAME", ""))
PY
)"
DB_ENGINE="$(printf '%s\n' "$DB_INFO" | sed -n '1p')"
DB_NAME="$(printf '%s\n' "$DB_INFO" | sed -n '2p')"
if [ "$DB_ENGINE" = "django.db.backends.sqlite3" ] && [ -n "$DB_NAME" ] && [ -f "$DB_NAME" ]; then
  DB_BACKUP="$BACKUP_DIR/db_${STAMP}.sqlite3"
  if command -v sqlite3 >/dev/null 2>&1; then
    sqlite3 "$DB_NAME" ".backup '$DB_BACKUP'"
  else
    cp -p "$DB_NAME" "$DB_BACKUP"
  fi
  log "SQLite backup created: $DB_BACKUP"
else
  log "Database is not a local SQLite file; skipping file backup (engine=$DB_ENGINE, name=$DB_NAME)"
fi

log "Migration plan"
python3 manage.py migrate --plan

log "Migrations"
python3 manage.py migrate

log "Static files"
python3 manage.py collectstatic --noinput

if [ "$RUN_TESTS" = "1" ]; then
  log "Tests"
  python3 manage.py test wind.tests.WindForecastModuleTests -v 2
else
  log "RUN_TESTS=0, skipping tests"
fi

log "Restarting web process"
if [ -f "$GUNICORN_PID_FILE_DEFAULT" ]; then
  sudo env GUNICORN_PID_FILE="$GUNICORN_PID_FILE_DEFAULT" bash deploy/restart_portal.sh
else
  sudo env GUNICORN_PORT="$GUNICORN_PORT_DEFAULT" PREFER_DIRECT_GUNICORN_RELOAD=1 bash deploy/restart_portal.sh
fi

log "Verifying deployed source marker"
rg -n "$EXPECTED_MARKER|data-postfactum-panel-version" wind/templates/wind/station_forecast_list.html

log "OK. Hard refresh browser (Ctrl+F5)."
