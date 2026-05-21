#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BRANCH="${BRANCH:-main}"
REMOTE="${REMOTE:-origin}"

ASSET_BACKUP_DIR="${ASSET_BACKUP_DIR:-$PROJECT_DIR/.local_assets_backup}"
GUIDE_SRC_DIR="$PROJECT_DIR/dashboard/static/dashboard/img/guide"
GUIDE_BACKUP_DIR="$ASSET_BACKUP_DIR/guide"

cd "$PROJECT_DIR"

mkdir -p "$GUIDE_BACKUP_DIR"

echo "=== A) Backup local guide assets before git reset ==="
if compgen -G "$GUIDE_SRC_DIR/*.svg" > /dev/null; then
  cp -f "$GUIDE_SRC_DIR"/*.svg "$GUIDE_BACKUP_DIR"/
  echo "[backup] Saved guide svg files to $GUIDE_BACKUP_DIR"
else
  echo "[backup] No local guide svg files found"
fi

echo "=== B) Update code from $REMOTE/$BRANCH ==="
git fetch --all --tags --prune
git checkout "$BRANCH"
git reset --hard "$REMOTE/$BRANCH"

echo "=== C) Restore preserved local guide assets ==="
if compgen -G "$GUIDE_BACKUP_DIR/*.svg" > /dev/null; then
  cp -f "$GUIDE_BACKUP_DIR"/*.svg "$GUIDE_SRC_DIR"/
  echo "[restore] Restored guide svg files from backup"
else
  echo "[restore] No backup guide assets to restore"
fi

echo "=== D) Django checks & deploy ==="
python3 manage.py makemigrations --check --dry-run
python3 manage.py migrate
python3 manage.py collectstatic --noinput
bash deploy/restart_portal.sh

echo "=== E) Smoke checks ==="
curl -I http://127.0.0.1:8000/ || true
curl -I "http://127.0.0.1:8000/api/assistant/query/" || true

echo "=== DONE ==="
git log -1 --oneline
