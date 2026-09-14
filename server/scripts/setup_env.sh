#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VQA_DEP_PROFILE="${OSL_VQA_DEP_PROFILE:-qwen}"

if [[ -z "${CONDA_DEFAULT_ENV:-}" && -z "${VIRTUAL_ENV:-}" ]]; then
  echo "No active Python environment detected."
  echo "Activate a Conda or virtual environment, then run this script again."
  exit 1
fi

if ! command -v python >/dev/null 2>&1; then
  echo "Python is not available in the active environment."
  exit 1
fi

python -c 'import sys; assert sys.version_info >= (3, 12), "Python 3.12 or newer is required"'
echo "Installing the server and its pinned OpenSportsLib release from PyPI"
if command -v uv >/dev/null 2>&1; then
  uv pip install --python "$(command -v python)" pip -e "${ROOT_DIR}"
else
  python -m pip install -e "${ROOT_DIR}"
fi

case "${VQA_DEP_PROFILE}" in
  qwen)
    echo "Running OpenSportsLib Qwen dependency setup"
    opensportslib setup --vqa_qwen
    ;;
  xvars)
    echo "Running OpenSportsLib X-VARS dependency setup"
    opensportslib setup --vqa_xvars
    ;;
  none)
    echo "Skipping OpenSportsLib VQA dependency override setup"
    ;;
  *)
    echo "Unsupported OSL_VQA_DEP_PROFILE='${VQA_DEP_PROFILE}'"
    echo "Use one of: qwen, xvars, none"
    exit 1
    ;;
esac

if command -v redis-server >/dev/null 2>&1; then
  echo "Redis server already available on PATH; skipping Conda Redis installation"
elif command -v conda >/dev/null 2>&1; then
  echo "Installing Redis server in Conda environment"
  conda install -y -c conda-forge redis
else
  echo "conda command not found; skipping Redis server install"
fi

echo "Generating flat standalone configs under ${ROOT_DIR}/config"
python "${ROOT_DIR}/scripts/generate_flat_configs.py"

if [[ ! -f "${ROOT_DIR}/.env" ]]; then
  cp "${ROOT_DIR}/.env.example" "${ROOT_DIR}/.env"
  echo "Created ${ROOT_DIR}/.env from .env.example"
else
  echo ".env already exists; leaving it unchanged"
fi

mkdir -p \
  "${ROOT_DIR}/runtime/jobs" \
  "${ROOT_DIR}/runtime/results" \
  "${ROOT_DIR}/runtime/sessions" \
  "${ROOT_DIR}/runtime/tmp" \
  "${ROOT_DIR}/logs"

echo
echo "Setup complete."
echo "Next steps:"
echo "1. Review ${ROOT_DIR}/.env; portable config paths and default Hugging Face model IDs are already set"
echo "   Active VQA dependency profile: ${VQA_DEP_PROFILE}"
echo "2. Start services: ${ROOT_DIR}/scripts/start_all.sh"
echo "   It reuses an existing Redis service or starts the project-managed Redis server"
echo "   Worker execution mode defaults to 'simple' via OSL_WORKER_EXECUTION_MODE"
