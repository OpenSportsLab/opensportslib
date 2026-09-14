#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="${ROOT_DIR}/.env"
ENV_EXAMPLE_FILE="${ROOT_DIR}/.env.example"

cd "${ROOT_DIR}"

if [[ ! -f "${ENV_FILE}" ]]; then
  if [[ ! -f "${ENV_EXAMPLE_FILE}" ]]; then
    echo "Cannot create .env: .env.example is missing."
    exit 1
  fi
  cp "${ENV_EXAMPLE_FILE}" "${ENV_FILE}"
  echo "Created .env from .env.example"
fi

if [[ "$#" -eq 0 ]]; then
  set -- up -d
fi

exec docker compose "$@"
