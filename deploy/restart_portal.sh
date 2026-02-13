#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/workspace/forecast_platform}"
SERVICE_NAME="${SERVICE_NAME:-gunicorn}"
SUPERVISOR_PROGRAM="${SUPERVISOR_PROGRAM:-forecast_portal}"
DOCKER_CONTAINER="${DOCKER_CONTAINER:-forecast_portal_web}"

cd "$PROJECT_DIR"

echo "[restart] Detecting process manager..."
if command -v systemctl >/dev/null 2>&1 && systemctl list-unit-files | rg -q "^${SERVICE_NAME}\.service"; then
  echo "[restart] Using systemd unit ${SERVICE_NAME}.service"
  sudo systemctl restart "${SERVICE_NAME}.service"
  sudo systemctl status "${SERVICE_NAME}.service" --no-pager --lines=20
  exit 0
fi

if command -v supervisorctl >/dev/null 2>&1 && supervisorctl status | rg -q "^${SUPERVISOR_PROGRAM}\b"; then
  echo "[restart] Using supervisor program ${SUPERVISOR_PROGRAM}"
  supervisorctl restart "${SUPERVISOR_PROGRAM}"
  supervisorctl status "${SUPERVISOR_PROGRAM}"
  exit 0
fi

if command -v docker >/dev/null 2>&1 && docker ps --format '{{.Names}}' | rg -q "^${DOCKER_CONTAINER}$"; then
  echo "[restart] Using docker container ${DOCKER_CONTAINER}"
  docker restart "${DOCKER_CONTAINER}"
  docker ps --filter "name=${DOCKER_CONTAINER}"
  exit 0
fi

echo "[restart] No known manager found (systemd/supervisor/docker)."
echo "[restart] Fallback: run Django dev server (not for production)."
python manage.py runserver 0.0.0.0:8000
