#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
API_HOST="${OSL_SERVER_HOST:-0.0.0.0}"
API_PORT="${OSL_SERVER_PORT:-8000}"
PID_DIR="${ROOT_DIR}/runtime/pids"
PID_FILE="${PID_DIR}/api.pid"
LOG_FILE="${ROOT_DIR}/logs/api.log"

mkdir -p "${PID_DIR}" "${ROOT_DIR}/logs"

if [[ -f "${PID_FILE}" ]]; then
  existing_pid="$(cat "${PID_FILE}")"
  if [[ -n "${existing_pid}" ]] && kill -0 "${existing_pid}" >/dev/null 2>&1; then
    echo "API already appears to be running with PID ${existing_pid}"
    echo "Log file: ${LOG_FILE}"
    exit 0
  fi
  rm -f "${PID_FILE}"
fi

cd "${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

nohup python -m uvicorn app.main:app \
  --host "${API_HOST}" \
  --port "${API_PORT}" \
  > "${LOG_FILE}" 2>&1 < /dev/null &

api_pid=$!
echo "${api_pid}" > "${PID_FILE}"

echo "API started in background on http://${API_HOST}:${API_PORT}"
echo "PID: ${api_pid}"
echo "Log file: ${LOG_FILE}"
