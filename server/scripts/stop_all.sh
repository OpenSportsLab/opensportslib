#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PID_DIR="${ROOT_DIR}/runtime/pids"
REDIS_PID_FILE="${ROOT_DIR}/runtime/redis/redis.pid"
API_PID_FILE="${PID_DIR}/api.pid"
WORKER_PID_FILE="${PID_DIR}/worker.pid"
stop_from_pid_file() {
  local name="$1"
  local pid_file="$2"

  if [[ ! -f "${pid_file}" ]]; then
    echo "${name}: no PID file found"
    return 0
  fi

  local pid
  pid="$(cat "${pid_file}")"
  if [[ -z "${pid}" ]]; then
    echo "${name}: PID file is empty"
    rm -f "${pid_file}"
    return 0
  fi

  if kill -0 "${pid}" >/dev/null 2>&1; then
    kill "${pid}"
    echo "${name}: stopped PID ${pid}"
  else
    echo "${name}: process ${pid} not running"
  fi

  rm -f "${pid_file}"
}

stop_from_pid_file "Worker" "${WORKER_PID_FILE}"
stop_from_pid_file "API" "${API_PID_FILE}"
stop_from_pid_file "Redis" "${REDIS_PID_FILE}"
"${ROOT_DIR}/scripts/clean_runtime.sh"

echo
echo "Stop commands completed."
