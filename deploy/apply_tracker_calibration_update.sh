#!/usr/bin/env bash
set -euo pipefail

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  cat <<'HELP'
Usage:
  bash deploy/apply_tracker_calibration_update.sh
  RUN_TESTS=0 bash deploy/apply_tracker_calibration_update.sh
  SKIP_SYNC=1 bash deploy/apply_tracker_calibration_update.sh
  TARGET_REF=origin/main bash deploy/apply_tracker_calibration_update.sh
  DIAG_STATION="Shu 100 MW" DIAG_FROM=2026-05-25 DIAG_TO=2026-05-31 bash deploy/apply_tracker_calibration_update.sh

Environment variables:
  PROJECT_DIR               Project path (default: ~/forecast_platform)
  VENV_PATH                 Virtualenv activate path (default: $PROJECT_DIR/venv/bin/activate)
  TARGET_REF                Ref to deploy after fetch (default: origin/main)
  SKIP_SYNC=1               Do not fetch/checkout/reset; deploy currently checked-out source
  STASH_LOCAL_CHANGES=1     Stash local changes instead of failing on a dirty worktree
  RUN_TESTS=0               Skip targeted Django tests
  BACKUP_DIR                Backup directory (default: $PROJECT_DIR/backups/tracker_calibration)
  DIAG_STATION              Optional station name for post-deploy operational/postfact diagnostics
  DIAG_STATION_ID           Optional station id for diagnostics (used instead of DIAG_STATION)
  DIAG_FROM                 Optional diagnostics start date, YYYY-MM-DD
  DIAG_TO                   Optional diagnostics end date, YYYY-MM-DD
  DIAG_FORECAST_SCOPE       Diagnostics operational scope (default: main)
  DIAG_POSTFACT_SCOPE       Diagnostics postfact scope (default: test)
  DIAG_THRESHOLD            Diagnostics warning threshold percent (default: 5)
  GUNICORN_PID_FILE_DEFAULT Gunicorn pid file path (default: /run/gunicorn.pid)
  GUNICORN_PORT_DEFAULT     Fallback app port for restart/smoke check (default: 8000)

Safety notes:
  - Saves git status and diffs before changing the checkout.
  - Fails on dirty worktree unless STASH_LOCAL_CHANGES=1 is set.
  - Creates a SQLite backup before migrations when the default DB is a local SQLite file.
  - Runs syntax checks, migration dry-run, migrate, collectstatic and a restart.
  - Optional diagnostics print the exact hourly columns requested for Shu-style tracker checks.
HELP
  exit 0
fi

PROJECT_DIR="${PROJECT_DIR:-$HOME/forecast_platform}"
VENV_PATH="${VENV_PATH:-$PROJECT_DIR/venv/bin/activate}"
TARGET_REF="${TARGET_REF:-origin/main}"
SKIP_SYNC="${SKIP_SYNC:-0}"
STASH_LOCAL_CHANGES="${STASH_LOCAL_CHANGES:-0}"
RUN_TESTS="${RUN_TESTS:-1}"
BACKUP_DIR="${BACKUP_DIR:-$PROJECT_DIR/backups/tracker_calibration}"
GUNICORN_PID_FILE_DEFAULT="${GUNICORN_PID_FILE_DEFAULT:-/run/gunicorn.pid}"
GUNICORN_PORT_DEFAULT="${GUNICORN_PORT_DEFAULT:-8000}"
EXPECTED_MARKER="compare_solar_forecast_modes"
STAMP="$(date +%Y%m%d_%H%M%S)"

log() {
  echo "[tracker-calibration-update] $*"
}

fail() {
  echo "[tracker-calibration-update] ERROR: $*" >&2
  exit 1
}

cd "$PROJECT_DIR"
[ -d .git ] || fail "not a git repository: $PROJECT_DIR"
mkdir -p "$BACKUP_DIR"

log "Project: $PROJECT_DIR"
log "Backups: $BACKUP_DIR"
log "Saving pre-update git state"
git status --short > "$BACKUP_DIR/git_status_before_${STAMP}.txt" || true
git diff > "$BACKUP_DIR/git_diff_before_${STAMP}.patch" || true
git diff --cached > "$BACKUP_DIR/git_diff_cached_before_${STAMP}.patch" || true

if [ -n "$(git status --porcelain)" ]; then
  if [ "$STASH_LOCAL_CHANGES" = "1" ]; then
    log "Dirty worktree detected; stashing local changes"
    git stash push --include-untracked -m "tracker-calibration-before-${STAMP}"
  else
    fail "local changes detected. Inspect git status, or rerun with STASH_LOCAL_CHANGES=1. Saved status/diff in $BACKUP_DIR"
  fi
fi

if [ "$SKIP_SYNC" = "1" ]; then
  log "SKIP_SYNC=1, keeping current checkout"
else
  log "Fetching and deploying $TARGET_REF"
  git fetch --all --tags --prune
  git checkout "$TARGET_REF"
  if [[ "$TARGET_REF" == origin/* ]]; then
    git reset --hard "$TARGET_REF"
  fi
fi

log "Current commit: $(git rev-parse --short HEAD) $(git log -1 --pretty=%s)"

if ! rg -q "$EXPECTED_MARKER" dashboard/management/commands/compare_solar_forecast_modes.py; then
  fail "expected marker '$EXPECTED_MARKER' not found. Merge/pull the tracker calibration PR first."
fi

[ -f "$VENV_PATH" ] || fail "virtualenv activation file not found: $VENV_PATH"
# shellcheck disable=SC1090
source "$VENV_PATH"

log "Syntax checks"
python3 -m py_compile \
  dashboard/services/forecast_engine.py \
  dashboard/services/forecast_diagnostics.py \
  dashboard/management/commands/compare_solar_forecast_modes.py \
  dashboard/tests.py

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

log "Collecting static files"
python3 manage.py collectstatic --noinput

if [ "$RUN_TESTS" = "1" ]; then
  log "Targeted tests"
  python3 manage.py test \
    dashboard.tests.SingleAxisTrackerPostProcessingTests \
    dashboard.tests.TrackerMiddayFloorTests \
    dashboard.tests.ForecastModeDiagnosticsTests \
    -v 2
else
  log "RUN_TESTS=0, skipping tests"
fi

log "Restarting web process"
if [ -f "$GUNICORN_PID_FILE_DEFAULT" ]; then
  sudo env GUNICORN_PID_FILE="$GUNICORN_PID_FILE_DEFAULT" bash deploy/restart_portal.sh
else
  sudo env GUNICORN_PORT="$GUNICORN_PORT_DEFAULT" PREFER_DIRECT_GUNICORN_RELOAD=1 bash deploy/restart_portal.sh
fi

if [ -n "${DIAG_STATION_ID:-}" ] || [ -n "${DIAG_STATION:-}" ]; then
  [ -n "${DIAG_FROM:-}" ] || fail "DIAG_FROM is required when diagnostics are enabled"
  [ -n "${DIAG_TO:-}" ] || fail "DIAG_TO is required when diagnostics are enabled"
  log "Running operational/postfact diagnostics"
  DIAG_ARGS=(--from "$DIAG_FROM" --to "$DIAG_TO" --forecast-scope "${DIAG_FORECAST_SCOPE:-main}" --postfact-scope "${DIAG_POSTFACT_SCOPE:-test}" --threshold "${DIAG_THRESHOLD:-5}")
  if [ -n "${DIAG_STATION_ID:-}" ]; then
    DIAG_ARGS=(--station-id "$DIAG_STATION_ID" "${DIAG_ARGS[@]}")
  else
    DIAG_ARGS=(--station "$DIAG_STATION" "${DIAG_ARGS[@]}")
  fi
  python3 manage.py compare_solar_forecast_modes "${DIAG_ARGS[@]}" | tee "$BACKUP_DIR/diagnostics_${STAMP}.txt"
  log "Diagnostics saved: $BACKUP_DIR/diagnostics_${STAMP}.txt"
fi

log "Smoke check"
curl -fsS -I "http://127.0.0.1:${GUNICORN_PORT_DEFAULT}/" || true

log "OK. HEAD=$(git rev-parse --short HEAD)"
