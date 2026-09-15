#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

if [[ "${OSL_PREDOWNLOAD_ON_START:-false}" == "true" ]]; then
  echo "Predownloading configured Hugging Face assets..."
  "${ROOT_DIR}/scripts/download_all_weights.sh"
fi

echo "Starting Redis..."
"${ROOT_DIR}/scripts/start_redis.sh"

echo "Starting API..."
"${ROOT_DIR}/scripts/start_api.sh"

echo "Starting worker..."
"${ROOT_DIR}/scripts/start_worker.sh"

echo
echo "All services start commands completed."
echo "Redis log:  ${ROOT_DIR}/logs/redis.log"
echo "API log:    ${ROOT_DIR}/logs/api.log"
echo "Worker log: ${ROOT_DIR}/logs/worker.log"
