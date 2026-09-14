#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PID_DIR="${ROOT_DIR}/runtime/pids"
PID_FILE="${PID_DIR}/worker.pid"
LOG_FILE="${ROOT_DIR}/logs/worker.log"

mkdir -p "${PID_DIR}" "${ROOT_DIR}/logs"

if [[ -f "${PID_FILE}" ]]; then
  existing_pid="$(cat "${PID_FILE}")"
  if [[ -n "${existing_pid}" ]] && kill -0 "${existing_pid}" >/dev/null 2>&1; then
    echo "Worker already appears to be running with PID ${existing_pid}"
    echo "Log file: ${LOG_FILE}"
    exit 0
  fi
  rm -f "${PID_FILE}"
fi

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

nohup python -m worker.main > "${LOG_FILE}" 2>&1 < /dev/null &

worker_pid=$!
echo "${worker_pid}" > "${PID_FILE}"

echo "Worker started in background"
echo "PID: ${worker_pid}"
echo "Log file: ${LOG_FILE}"
