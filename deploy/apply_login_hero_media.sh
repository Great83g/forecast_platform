#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash deploy/apply_login_hero_media.sh /tmp/login-hero.mp4 /tmp/login-background.png
# Optional env:
#   PROJECT_DIR=/workspace/forecast_platform
#   RESTART=1 (default) | RESTART=0

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MEDIA_DIR="${MEDIA_DIR:-$PROJECT_DIR/media}"
RESTART="${RESTART:-1}"

VIDEO_SRC="${1:-}"
POSTER_SRC="${2:-}"

if [[ -z "$VIDEO_SRC" || -z "$POSTER_SRC" ]]; then
  echo "Usage: bash deploy/apply_login_hero_media.sh <video.mp4> <poster.png>"
  exit 1
fi

if [[ "$VIDEO_SRC" == *"/полный/путь/"* || "$POSTER_SRC" == *"/полный/путь/"* ]]; then
  echo "[hero-media] ERROR: you passed a placeholder path '/полный/путь/...'."
  echo "[hero-media] Replace it with real server paths, for example:"
  echo "  bash deploy/apply_login_hero_media.sh ~/Downloads/login-hero.mp4 ~/Downloads/login-background.png"
  exit 1
fi

if [[ ! -f "$VIDEO_SRC" ]]; then
  echo "[hero-media] ERROR: video not found: $VIDEO_SRC"
  exit 1
fi

if [[ ! -f "$POSTER_SRC" ]]; then
  echo "[hero-media] ERROR: poster not found: $POSTER_SRC"
  exit 1
fi

mkdir -p "$MEDIA_DIR"
cp "$VIDEO_SRC" "$MEDIA_DIR/login-hero.mp4"
cp "$POSTER_SRC" "$MEDIA_DIR/login-background.png"

echo "[hero-media] Copied video -> $MEDIA_DIR/login-hero.mp4"
echo "[hero-media] Copied poster -> $MEDIA_DIR/login-background.png"
ls -lh "$MEDIA_DIR/login-hero.mp4" "$MEDIA_DIR/login-background.png"

if [[ "$RESTART" == "1" ]]; then
  echo "[hero-media] Restarting portal..."
  bash "$PROJECT_DIR/deploy/restart_portal.sh"
else
  echo "[hero-media] RESTART=0, skip restart"
fi

echo "[hero-media] Done. Check: https://intech-forecast.com/login/"
