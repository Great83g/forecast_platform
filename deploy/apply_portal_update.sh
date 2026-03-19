#!/usr/bin/env bash
set -euo pipefail

# Простой прод-апдейт портала (как в ручных командах).
PROJECT_DIR="${PROJECT_DIR:-$HOME/forecast_platform}"
VENV_PATH="${VENV_PATH:-venv/bin/activate}"
GUNICORN_PID_FILE_DEFAULT="${GUNICORN_PID_FILE_DEFAULT:-/run/gunicorn.pid}"
GUNICORN_PORT_DEFAULT="${GUNICORN_PORT_DEFAULT:-8000}"
STASH_LOCAL_CHANGES="${STASH_LOCAL_CHANGES:-0}"
AUTO_STASH_MESSAGE="${AUTO_STASH_MESSAGE:-manual-before-update}"

cd "$PROJECT_DIR"

if [ ! -d .git ]; then
  echo "[portal-update] ERROR: not a git repository: $PROJECT_DIR"
  exit 1
fi

if [ -n "$(git status --porcelain)" ]; then
  if [ "$STASH_LOCAL_CHANGES" = "1" ]; then
    echo "[portal-update] dirty worktree detected; stashing local changes"
    git stash push --include-untracked -m "$AUTO_STASH_MESSAGE"
  else
    echo "[portal-update] ERROR: local unstaged/uncommitted changes detected"
    echo "[portal-update] Run one of the following:"
    echo "[portal-update]   git status --short"
    echo "[portal-update]   git stash push --include-untracked -m 'manual-before-update'"
    echo "[portal-update] or rerun with: STASH_LOCAL_CHANGES=1 bash deploy/apply_portal_update.sh"
    exit 1
  fi
fi

echo "[portal-update] git status --short"
git status --short

echo "[portal-update] git pull --rebase"
git pull --rebase

if [ ! -f "$VENV_PATH" ]; then
  echo "[portal-update] ERROR: virtualenv activation file not found: $VENV_PATH"
  echo "[portal-update] Set VENV_PATH, for example: VENV_PATH=/opt/venv/bin/activate"
  exit 1
fi

# shellcheck disable=SC1090
source "$VENV_PATH"

echo "[portal-update] python3 manage.py migrate"
python3 manage.py migrate

echo "[portal-update] python3 manage.py collectstatic --noinput"
python3 manage.py collectstatic --noinput

if [ -f "$GUNICORN_PID_FILE_DEFAULT" ]; then
  echo "[portal-update] restart via pid file: $GUNICORN_PID_FILE_DEFAULT"
  GUNICORN_PID_FILE="$GUNICORN_PID_FILE_DEFAULT" bash deploy/restart_portal.sh
else
  echo "[portal-update] pid file not found, restart via port: $GUNICORN_PORT_DEFAULT"
  GUNICORN_PORT="$GUNICORN_PORT_DEFAULT" bash deploy/restart_portal.sh
fi

echo "[portal-update] smoke check"
curl -I "http://127.0.0.1:${GUNICORN_PORT_DEFAULT}/" || true

echo "[portal-update] done; HEAD=$(git rev-parse --short HEAD)"
