#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SERVICE_CANDIDATES="${SERVICE_CANDIDATES:-gunicorn forecast-platform forecast_portal backend}"
SUPERVISOR_PROGRAM="${SUPERVISOR_PROGRAM:-forecast_portal}"
DOCKER_CONTAINER="${DOCKER_CONTAINER:-forecast_portal_web}"
ALLOW_RUNSERVER_FALLBACK="${ALLOW_RUNSERVER_FALLBACK:-0}"
GUNICORN_PORT_EXPLICIT="${GUNICORN_PORT:-}"
GUNICORN_PORT="${GUNICORN_PORT:-8000}"
GUNICORN_PID_FILE="${GUNICORN_PID_FILE:-}"
GUNICORN_MATCH="${GUNICORN_MATCH:-gunicorn}"
SKIP_CONFLICT_CHECK="${SKIP_CONFLICT_CHECK:-0}"
PREFER_DIRECT_GUNICORN_RELOAD="${PREFER_DIRECT_GUNICORN_RELOAD:-auto}"

cd "$PROJECT_DIR"

echo "[restart] Detecting process manager..."

if [ "$SKIP_CONFLICT_CHECK" != "1" ] && command -v git >/dev/null 2>&1; then
  if git -C "$PROJECT_DIR" grep -nE '^(<<<<<<< |=======|>>>>>>>)' -- '*.py' '*.html' '*.js' '*.css' >/tmp/restart_portal_conflicts.txt 2>/dev/null; then
    echo "[restart] ERROR: unresolved merge conflict markers found in working tree."
    cat /tmp/restart_portal_conflicts.txt
    echo "[restart] Resolve conflicts first or run with SKIP_CONFLICT_CHECK=1 to bypass."
    exit 1
  fi
fi

restart_systemd() {
  local unit="$1"
  echo "[restart] Using systemd unit ${unit}.service"
  systemctl restart "${unit}.service"
  systemctl --no-pager --lines=20 status "${unit}.service"
}

restart_systemd_user() {
  local unit="$1"
  echo "[restart] Using user systemd unit ${unit}.service"
  systemctl --user restart "${unit}.service"
  systemctl --user --no-pager --lines=20 status "${unit}.service"
}


prefer_direct_gunicorn_reload() {
  case "$PREFER_DIRECT_GUNICORN_RELOAD" in
    1|true|yes|on)
      return 0
      ;;
    0|false|no|off)
      return 1
      ;;
  esac

  if [ -n "$GUNICORN_PID_FILE" ] || [ -n "$GUNICORN_PORT_EXPLICIT" ]; then
    return 0
  fi

  return 1
}

reload_gunicorn_master() {
  local master_pid
  local listener_pid

  if [ -n "$GUNICORN_PID_FILE" ] && [ -f "$GUNICORN_PID_FILE" ]; then
    master_pid="$(cat "$GUNICORN_PID_FILE" 2>/dev/null || true)"
    if [ -n "$master_pid" ] && kill -0 "$master_pid" 2>/dev/null; then
      echo "[restart] Using gunicorn pid from GUNICORN_PID_FILE=${GUNICORN_PID_FILE}: ${master_pid}"
      kill -HUP "$master_pid"
      ps -fp "$master_pid"
      return 0
    fi
  fi

  # Debug list helps when process titles differ between setups.
  pgrep -af 'gunicorn' >/tmp/restart_portal_gunicorn_ps.txt || true
  if [ -s /tmp/restart_portal_gunicorn_ps.txt ]; then
    echo "[restart] Detected gunicorn-related processes:"
    cat /tmp/restart_portal_gunicorn_ps.txt
  fi

  # 1) Standard gunicorn title: "gunicorn: master [...]"
  master_pid="$(pgrep -o -f 'gunicorn: master' || true)"

  # 2) Some installs expose only the launch command, e.g. ".../bin/gunicorn backend.wsgi"
  if [ -z "$master_pid" ]; then
    master_pid="$(pgrep -o -f 'gunicorn .*\.(wsgi|asgi)' || true)"
  fi

  # 3) Final fallback: plain executable name
  if [ -z "$master_pid" ]; then
    master_pid="$(pgrep -o -x gunicorn || true)"
  fi

  # 4) Port fallback: process listening on configured app port.
  if [ -z "$master_pid" ] && command -v ss >/dev/null 2>&1; then
    listener_pid="$(ss -ltnp 2>/dev/null | awk -v p=":${GUNICORN_PORT}" '$4 ~ p {print $NF}' | sed -n 's/.*pid=\([0-9]\+\).*/\1/p' | head -n1)"
    if [ -n "$listener_pid" ]; then
      if ps -p "$listener_pid" -o cmd= | rg -qi "$GUNICORN_MATCH"; then
        master_pid="$listener_pid"
      else
        # Some launchers wrap gunicorn; try parent pid.
        parent_pid="$(ps -o ppid= -p "$listener_pid" | awk '{print $1}' || true)"
        if [ -n "$parent_pid" ] && ps -p "$parent_pid" -o cmd= | rg -qi "$GUNICORN_MATCH"; then
          master_pid="$parent_pid"
        fi
      fi
    fi
  fi

  if [ -n "$master_pid" ]; then
    echo "[restart] Found gunicorn master/candidate process (${master_pid}), sending HUP for graceful reload"
    kill -HUP "$master_pid"
    ps -fp "$master_pid"
    return 0
  fi

  return 1
}

if prefer_direct_gunicorn_reload; then
  echo "[restart] Explicit gunicorn reload requested; skipping systemd/supervisor/docker detection"
  if reload_gunicorn_master; then
    exit 0
  fi
  echo "[restart] Direct gunicorn reload requested but no matching process was found; falling back to manager detection"
fi

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

if command -v systemctl >/dev/null 2>&1; then
  for service in $SERVICE_CANDIDATES; do
    if systemctl --user list-unit-files --type=service 2>/dev/null | rg -q "^${service}\.service"; then
      restart_systemd_user "$service"
      exit 0
    fi
  done
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

if reload_gunicorn_master; then
  exit 0
fi

echo "[restart] No known manager found (systemd/supervisor/docker)."
if [ "$ALLOW_RUNSERVER_FALLBACK" = "1" ]; then
  echo "[restart] Fallback enabled -> starting Django runserver (dev only)."
  exec python manage.py runserver 0.0.0.0:8000
fi

echo "[restart] Refusing to start runserver by default in production helper."
echo "[restart] Set ALLOW_RUNSERVER_FALLBACK=1 only for temporary diagnostics."
echo "[restart] Tip: export GUNICORN_PID_FILE=/path/to/gunicorn.pid or GUNICORN_PORT=8000"
exit 1
