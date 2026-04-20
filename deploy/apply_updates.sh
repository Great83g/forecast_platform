#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$HOME/forecast_platform}"
VENV_PATH="${VENV_PATH:-venv/bin/activate}"
GUNICORN_PID_FILE_DEFAULT="${GUNICORN_PID_FILE_DEFAULT:-/run/gunicorn.pid}"
GUNICORN_PORT_DEFAULT="${GUNICORN_PORT_DEFAULT:-8000}"
AUTO_STASH="${AUTO_STASH:-0}"
AUTO_STASH_MESSAGE="${AUTO_STASH_MESSAGE:-manual-before-update}"

cd "$PROJECT_DIR"

if [ ! -d .git ]; then
  echo "[deploy] ERROR: not a git repository: $PROJECT_DIR"
  exit 1
fi

echo "[deploy] Running update sequence:"
echo "[deploy]   cd $PROJECT_DIR"
echo "[deploy]   git status --short"
echo "[deploy]   git stash push --include-untracked -m '$AUTO_STASH_MESSAGE'   # optional when worktree is dirty"
echo "[deploy]   git pull --rebase"
echo "[deploy]   source $VENV_PATH"
echo "[deploy]   python3 manage.py migrate"
echo "[deploy]   python3 manage.py cleanup_model_cache"
echo "[deploy]   python3 manage.py collectstatic --noinput"

if [ -n "$(git status --porcelain)" ]; then
  if [ "$AUTO_STASH" = "1" ]; then
    echo "[deploy] dirty worktree detected; stashing local changes"
    git stash push --include-untracked -m "$AUTO_STASH_MESSAGE"
  else
    echo "[deploy] ERROR: local unstaged/uncommitted changes detected"
    echo "[deploy] Run:"
    echo "[deploy]   git status --short"
    echo "[deploy]   git stash push --include-untracked -m 'manual-before-update'"
    echo "[deploy] or rerun with: AUTO_STASH=1 bash deploy/apply_updates.sh"
    exit 1
  fi
fi

echo "[deploy] git status --short"
git status --short

echo "[deploy] Pulling latest changes (rebase)..."
git pull --rebase

if [ ! -f "$VENV_PATH" ]; then
  echo "[deploy] ERROR: virtualenv activation file not found: $VENV_PATH"
  echo "[deploy] Set VENV_PATH explicitly, for example: VENV_PATH=/opt/venv/bin/activate"
  exit 1
fi

echo "[deploy] Activating virtualenv: $VENV_PATH"
# shellcheck disable=SC1090
source "$VENV_PATH"

echo "[deploy] Planned migrations..."
python3 manage.py migrate --plan

echo "[deploy] Applying migrations..."
python3 manage.py migrate

echo "[deploy] Normalizing model cache..."
python3 manage.py cleanup_model_cache

echo "[deploy] Collecting static files..."
python3 manage.py collectstatic --noinput

if [ -f "$GUNICORN_PID_FILE_DEFAULT" ]; then
  echo "[deploy] Restarting via pid file: $GUNICORN_PID_FILE_DEFAULT"
  GUNICORN_PID_FILE="$GUNICORN_PID_FILE_DEFAULT" bash deploy/restart_portal.sh
else
  echo "[deploy] PID file not found, restarting via port: $GUNICORN_PORT_DEFAULT"
  GUNICORN_PORT="$GUNICORN_PORT_DEFAULT" bash deploy/restart_portal.sh
fi

echo "[deploy] Smoke check:"
curl -I "http://127.0.0.1:${GUNICORN_PORT_DEFAULT}/" || true

echo "[deploy] Done. Current commit: $(git rev-parse --short HEAD)"
