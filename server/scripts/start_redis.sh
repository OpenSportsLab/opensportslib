#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REDIS_PORT="${REDIS_PORT:-6379}"
REDIS_HOST="${REDIS_HOST:-127.0.0.1}"
REDIS_DATA_DIR="${ROOT_DIR}/runtime/redis"
REDIS_LOG_FILE="${ROOT_DIR}/logs/redis.log"
REDIS_PID_FILE="${REDIS_DATA_DIR}/redis.pid"

mkdir -p "${REDIS_DATA_DIR}" "${ROOT_DIR}/logs"

if command -v redis-cli >/dev/null 2>&1 && \
  [[ "$(redis-cli --raw -h "${REDIS_HOST}" -p "${REDIS_PORT}" ping 2>/dev/null || true)" == "PONG" ]]; then
  echo "Redis is already responding on ${REDIS_HOST}:${REDIS_PORT}; reusing it"
  exit 0
fi

if ! command -v redis-server >/dev/null 2>&1; then
  echo "redis-server is not installed or not on PATH."
  echo "If you use Conda, run: conda activate osl && conda install -c conda-forge redis"
  exit 1
fi

if [[ -f "${REDIS_PID_FILE}" ]]; then
  existing_pid="$(cat "${REDIS_PID_FILE}")"
  if [[ -n "${existing_pid}" ]] && kill -0 "${existing_pid}" >/dev/null 2>&1; then
    echo "Redis already appears to be running with PID ${existing_pid}"
    echo "Check with: redis-cli -h ${REDIS_HOST} -p ${REDIS_PORT} ping"
    exit 0
  fi
  rm -f "${REDIS_PID_FILE}"
fi

redis-server \
  --bind "${REDIS_HOST}" \
  --port "${REDIS_PORT}" \
  --dir "${REDIS_DATA_DIR}" \
  --daemonize yes \
  --logfile "${REDIS_LOG_FILE}" \
  --pidfile "${REDIS_PID_FILE}"

echo "Redis started on ${REDIS_HOST}:${REDIS_PORT}"
echo "Log file: ${REDIS_LOG_FILE}"
echo "Verify with: redis-cli -h ${REDIS_HOST} -p ${REDIS_PORT} ping"
