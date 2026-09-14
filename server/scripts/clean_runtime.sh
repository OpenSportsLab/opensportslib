#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RUNTIME_DIR="${ROOT_DIR}/runtime"
had_error=0

clear_dir_contents() {
  local dir_path="$1"
  local label="$2"

  if [[ -d "${dir_path}" ]]; then
    if find "${dir_path}" -mindepth 1 -maxdepth 1 ! -name '.gitkeep' -exec rm -rf {} +; then
      echo "${label} cleared: ${dir_path}"
    else
      echo "${label} cleanup hit permission errors: ${dir_path}"
      had_error=1
    fi
  else
    echo "${label} directory not found: ${dir_path}"
  fi
}

clear_dir_contents "${RUNTIME_DIR}/tmp" "Temp runtime"
clear_dir_contents "${RUNTIME_DIR}/jobs" "Job metadata"
clear_dir_contents "${RUNTIME_DIR}/results" "Job results"
clear_dir_contents "${RUNTIME_DIR}/sessions" "Session state"
clear_dir_contents "${RUNTIME_DIR}/redis" "Redis runtime"
clear_dir_contents "${RUNTIME_DIR}/pids" "PID files"

if [[ "${had_error}" -ne 0 ]]; then
  echo
  echo "Some runtime files could not be removed from the host shell."
  echo "If these came from older Docker runs, use: ./scripts/docker_stop_all.sh"
  exit 1
fi
