#!/usr/bin/env bash
set -euo pipefail

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  cat <<'HELP'
Usage:
  bash deploy/force_sync_and_restart.sh

Environment variables:
  PROJECT_DIR                Project path (default: repo root)
  TARGET_REMOTE              Git remote name to sync from (default: origin)
  TARGET_BRANCH              Branch to hard-reset to (default: main)
  VENV_PATH                  Virtualenv activate path (default: <repo>/venv/bin/activate)
  GUNICORN_PID_FILE_DEFAULT  Gunicorn pid file path (default: /run/gunicorn.pid)
  GUNICORN_PORT_DEFAULT      Fallback app port for restart helper (default: 8000)

This helper intentionally discards local changes via:
  git fetch --all --prune
  git checkout <branch>
  git reset --hard <remote>/<branch>
  git clean -fd
HELP
  exit 0
fi

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
TARGET_REMOTE="${TARGET_REMOTE:-origin}"
TARGET_BRANCH="${TARGET_BRANCH:-main}"
VENV_PATH="${VENV_PATH:-$PROJECT_DIR/venv/bin/activate}"
GUNICORN_PID_FILE_DEFAULT="${GUNICORN_PID_FILE_DEFAULT:-/run/gunicorn.pid}"
GUNICORN_PORT_DEFAULT="${GUNICORN_PORT_DEFAULT:-8000}"
TARGET_REF="${TARGET_REMOTE}/${TARGET_BRANCH}"

cd "$PROJECT_DIR"

if [ ! -d .git ]; then
  echo "[force-sync] ERROR: not a git repository: $PROJECT_DIR"
  exit 1
fi

if ! git remote get-url "$TARGET_REMOTE" >/dev/null 2>&1; then
  echo "[force-sync] ERROR: git remote not found: $TARGET_REMOTE"
  exit 1
fi

echo "[force-sync] Project: $PROJECT_DIR"
echo "[force-sync] Sync target: $TARGET_REF"
echo "[force-sync] WARNING: local changes and untracked files will be deleted"

echo "[force-sync] Fetching latest refs..."
git fetch --all --prune

echo "[force-sync] Aborting unfinished git ops (if any)..."
git rebase --abort >/dev/null 2>&1 || true
git merge --abort >/dev/null 2>&1 || true

echo "[force-sync] Checking out $TARGET_BRANCH"
git checkout "$TARGET_BRANCH"

echo "[force-sync] Resetting worktree to $TARGET_REF"
git reset --hard "$TARGET_REF"

echo "[force-sync] Cleaning untracked files"
git clean -fd

echo "[force-sync] Current HEAD: $(git rev-parse --short HEAD)"

if [ ! -f "$VENV_PATH" ]; then
  echo "[force-sync] ERROR: virtualenv activation file not found: $VENV_PATH"
  echo "[force-sync] Set VENV_PATH, for example: VENV_PATH=/opt/venv/bin/activate"
  exit 1
fi

echo "[force-sync] Activating virtualenv: $VENV_PATH"
# shellcheck disable=SC1090
source "$VENV_PATH"

echo "[force-sync] Running migrations..."
python3 manage.py migrate

echo "[force-sync] Collecting static files..."
python3 manage.py collectstatic --noinput

if [ -f "$GUNICORN_PID_FILE_DEFAULT" ]; then
  echo "[force-sync] Restarting via pid file: $GUNICORN_PID_FILE_DEFAULT"
  GUNICORN_PID_FILE="$GUNICORN_PID_FILE_DEFAULT" bash deploy/restart_portal.sh
else
  echo "[force-sync] PID file not found, restarting via port: $GUNICORN_PORT_DEFAULT"
  GUNICORN_PORT="$GUNICORN_PORT_DEFAULT" bash deploy/restart_portal.sh
fi

echo "[force-sync] Smoke check"
curl -I "http://127.0.0.1:${GUNICORN_PORT_DEFAULT}/" || true

echo "[force-sync] Done. HEAD=$(git rev-parse --short HEAD)"
