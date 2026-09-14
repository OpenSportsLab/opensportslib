#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

cd "${ROOT_DIR}"

docker compose down

docker compose run --rm --no-deps -u 0:0 --entrypoint bash api -lc "cd /app && ./scripts/clean_runtime.sh"

echo
echo "Docker services stopped and runtime cleaned."
