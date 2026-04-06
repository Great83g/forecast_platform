#!/usr/bin/env bash
# Usage: bash deploy/verify_update.sh
set -u

cd ~/forecast_platform || exit 1

if [ -f venv/bin/activate ]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
else
  echo "[ERR] venv/bin/activate not found"
  exit 1
fi

echo "=== VERIFY UPDATE START $(date -u '+%Y-%m-%d %H:%M:%S UTC') ==="

echo
echo "[1/6] Branch/HEAD"
git rev-parse --abbrev-ref HEAD
git rev-parse --short HEAD
git status --short

echo
echo "[2/6] Last commits"
git log --oneline -n 5

echo
echo "[3/6] Django migration status"
python3 manage.py showmigrations wind | tail -n 20 || true
if python3 manage.py showmigrations wind 2>/dev/null | rg -q '^\s*\[\s\]'; then
  echo "[WARN] Есть неприменённые миграции в app 'wind'. Запусти: python3 manage.py migrate"
fi

echo
echo "[4/6] Check pending model changes"
python3 manage.py makemigrations --check --dry-run || true

echo
echo "[5/6] Gunicorn processes"
pgrep -af "gunicorn.*backend.wsgi" || echo "gunicorn process not found"

echo
echo "[6/6] Port 8000 listener"
if command -v ss >/dev/null 2>&1; then
  ss -ltnp | rg ":8000" || echo "port 8000 not listening"
else
  echo "ss command not found"
fi

echo
echo "=== VERIFY UPDATE DONE ==="
