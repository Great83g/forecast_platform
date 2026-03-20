#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: bash deploy/apply_commit.sh <commit-or-tag-or-branch>"
  echo "Example: bash deploy/apply_commit.sh 96bc888"
  exit 2
fi

TARGET_REF="$1"
PROJECT_DIR="${PROJECT_DIR:-$HOME/forecast_platform}"
VENV_PATH="${VENV_PATH:-venv/bin/activate}"
GUNICORN_PID_FILE_DEFAULT="${GUNICORN_PID_FILE_DEFAULT:-/run/gunicorn.pid}"
GUNICORN_PORT_DEFAULT="${GUNICORN_PORT_DEFAULT:-8000}"
STASH_LOCAL_CHANGES="${STASH_LOCAL_CHANGES:-0}"
AUTO_STASH_MESSAGE="${AUTO_STASH_MESSAGE:-manual-before-apply-commit}"

cd "$PROJECT_DIR"

echo "[apply-commit] project: $PROJECT_DIR"
echo "[apply-commit] target:  $TARGET_REF"

if [ ! -d .git ]; then
  echo "[apply-commit] ERROR: not a git repo: $PROJECT_DIR"
  exit 1
fi

if [ -n "$(git status --porcelain)" ]; then
  if [ "$STASH_LOCAL_CHANGES" = "1" ]; then
    echo "[apply-commit] dirty worktree detected; stashing local changes"
    git stash push --include-untracked -m "$AUTO_STASH_MESSAGE"
  else
    echo "[apply-commit] ERROR: local unstaged/uncommitted changes detected"
    echo "[apply-commit] Run one of the following:"
    echo "[apply-commit]   git status --short"
    echo "[apply-commit]   git stash push --include-untracked -m 'manual-before-apply-commit'"
    echo "[apply-commit] or rerun with: STASH_LOCAL_CHANGES=1 bash deploy/apply_commit.sh <commit>"
    exit 1
  fi
fi

echo "[apply-commit] Fetching origin..."
git fetch --all --tags --prune

if ! git rev-parse --verify --quiet "$TARGET_REF^{commit}" >/dev/null; then
  if git rev-parse --verify --quiet "origin/$TARGET_REF^{commit}" >/dev/null; then
    TARGET_REF="origin/$TARGET_REF"
  else
    echo "[apply-commit] ERROR: ref not found: $1"
    exit 1
  fi
fi

echo "[apply-commit] Checking out $TARGET_REF"
git checkout "$TARGET_REF"

if [ ! -f "$VENV_PATH" ]; then
  echo "[apply-commit] ERROR: virtualenv activation file not found: $VENV_PATH"
  echo "[apply-commit] Set VENV_PATH, for example: VENV_PATH=/opt/venv/bin/activate"
  exit 1
fi

# shellcheck disable=SC1090
source "$VENV_PATH"

echo "[apply-commit] Planned migrations..."
python3 manage.py migrate --plan

echo "[apply-commit] Applying migrations..."
python3 manage.py migrate

echo "[apply-commit] Normalizing model cache..."
python3 manage.py cleanup_model_cache

echo "[apply-commit] Collecting static files..."
python3 manage.py collectstatic --noinput

if [ -f "$GUNICORN_PID_FILE_DEFAULT" ]; then
  echo "[apply-commit] Restarting portal via PID file: $GUNICORN_PID_FILE_DEFAULT"
  GUNICORN_PID_FILE="$GUNICORN_PID_FILE_DEFAULT" bash deploy/restart_portal.sh
else
  echo "[apply-commit] PID file not found, restarting via port: $GUNICORN_PORT_DEFAULT"
  GUNICORN_PORT="$GUNICORN_PORT_DEFAULT" bash deploy/restart_portal.sh
fi

echo "[apply-commit] Smoke check"
curl -I "http://127.0.0.1:${GUNICORN_PORT_DEFAULT}/" || true

echo "[apply-commit] Done. HEAD=$(git rev-parse --short HEAD)"
