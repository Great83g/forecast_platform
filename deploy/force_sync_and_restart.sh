#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
TARGET_REMOTE="${TARGET_REMOTE:-origin}"
TARGET_BRANCH="${TARGET_BRANCH:-main}"
VENV_PATH="${VENV_PATH:-$PROJECT_DIR/venv/bin/activate}"
GUNICORN_PID_FILE_DEFAULT="${GUNICORN_PID_FILE_DEFAULT:-/run/gunicorn.pid}"
GUNICORN_PORT_DEFAULT="${GUNICORN_PORT_DEFAULT:-8000}"

cd "$PROJECT_DIR"

echo "[force-sync] Fetching latest refs..."
git fetch "$TARGET_REMOTE" --prune

echo "[force-sync] Aborting unfinished git ops (if any)..."
git rebase --abort >/dev/null 2>&1 || true
git merge --abort >/dev/null 2>&1 || true

echo "[force-sync] Resetting worktree to ${TARGET_REMOTE}/${TARGET_BRANCH}"
git checkout "$TARGET_BRANCH"
git reset --hard "${TARGET_REMOTE}/${TARGET_BRANCH}"
git clean -fd

echo "[force-sync] Activating virtualenv: $VENV_PATH"
# shellcheck disable=SC1090
source "$VENV_PATH"

echo "[force-sync] Running migrations..."
python3 manage.py migrate

if [ -f "$GUNICORN_PID_FILE_DEFAULT" ]; then
  echo "[force-sync] Restarting via pid file: $GUNICORN_PID_FILE_DEFAULT"
  GUNICORN_PID_FILE="$GUNICORN_PID_FILE_DEFAULT" bash deploy/restart_portal.sh
else
  echo "[force-sync] Restarting via port: $GUNICORN_PORT_DEFAULT"
  GUNICORN_PORT="$GUNICORN_PORT_DEFAULT" bash deploy/restart_portal.sh
fi

echo "[force-sync] Done."
