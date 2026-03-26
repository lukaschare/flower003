#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT_DEFAULT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$PROJECT_ROOT_DEFAULT}"
STATE_FILE="$PROJECT_ROOT/.run_state/current_run.env"
COMPOSE_FILE="$PROJECT_ROOT/docker/docker-compose.yml"
SUDO_DOCKER=0
PRUNE=0

usage() {
  cat <<USAGE
Usage:
  $(basename "$0") [--project-root PATH] [--sudo-docker] [--prune]

Options:
  --sudo-docker   Use sudo for docker compose down.
  --prune         Also run docker image prune -f after shutdown.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --project-root)
      PROJECT_ROOT="$2"; shift 2 ;;
    --sudo-docker)
      SUDO_DOCKER=1; shift ;;
    --prune)
      PRUNE=1; shift ;;
    --help|-h)
      usage; exit 0 ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1 ;;
  esac
done

DOCKER_BIN="docker"
if [[ "$SUDO_DOCKER" == "1" ]]; then
  DOCKER_BIN="sudo docker"
fi

if [[ -f "$STATE_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$STATE_FILE"
fi

if [[ -f "$COMPOSE_FILE" ]]; then
  echo "[stop] docker compose down"
  (cd "$PROJECT_ROOT" && bash -lc "$DOCKER_BIN compose -f docker/docker-compose.yml down") || true
fi

echo "[stop] killing host-side processes"
pkill -f "veins_launchd -vv -c sumo" || true
pkill -f "veins_launchd -vv -c sumo-gui" || true
pkill -f "fedits_veins_rsu -u Cmdenv -c Default" || true
pkill -f "streamlit run fl/dashboard_streamlit.py" || true

if [[ -n "${SUMO_LAUNCHD_PID:-}" ]]; then kill "$SUMO_LAUNCHD_PID" 2>/dev/null || true; fi
if [[ -n "${VEINS_PID:-}" ]]; then kill "$VEINS_PID" 2>/dev/null || true; fi
if [[ -n "${DASHBOARD_PID:-}" ]]; then kill "$DASHBOARD_PID" 2>/dev/null || true; fi

if [[ "$PRUNE" == "1" ]]; then
  echo "[stop] docker image prune -f"
  bash -lc "$DOCKER_BIN image prune -f" || true
fi

echo "[stop] done"
