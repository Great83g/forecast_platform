#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/workspace/forecast_platform}"
SERVICE_CANDIDATES="${SERVICE_CANDIDATES:-gunicorn forecast-platform forecast_portal backend}"
SUPERVISOR_PROGRAM="${SUPERVISOR_PROGRAM:-forecast_portal}"
DOCKER_CONTAINER="${DOCKER_CONTAINER:-forecast_portal_web}"
ALLOW_RUNSERVER_FALLBACK="${ALLOW_RUNSERVER_FALLBACK:-0}"

cd "$PROJECT_DIR"

echo "[restart] Detecting process manager..."

restart_systemd() {
  local unit="$1"
  echo "[restart] Using systemd unit ${unit}.service"
  systemctl restart "${unit}.service"
  systemctl --no-pager --lines=20 status "${unit}.service"
}

if command -v systemctl >/dev/null 2>&1; then
  for service in $SERVICE_CANDIDATES; do
    if systemctl list-unit-files --type=service | rg -q "^${service}\.service"; then
      restart_systemd "$service"
      exit 0
    fi
  done

  # fallback: first running gunicorn-like service
  running_service="$(systemctl list-units --type=service --state=running --no-legend | awk '{print $1}' | rg 'gunicorn|uvicorn|daphne' | head -n1 || true)"
  if [ -n "$running_service" ]; then
    service_name="${running_service%.service}"
    restart_systemd "$service_name"
    exit 0
  fi
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
if [ "$ALLOW_RUNSERVER_FALLBACK" = "1" ]; then
  echo "[restart] Fallback enabled -> starting Django runserver (dev only)."
  exec python manage.py runserver 0.0.0.0:8000
fi

echo "[restart] Refusing to start runserver by default in production helper."
echo "[restart] Set ALLOW_RUNSERVER_FALLBACK=1 only for temporary diagnostics."
exit 1
