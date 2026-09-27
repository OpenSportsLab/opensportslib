#!/usr/bin/env bash
set -euo pipefail

# Keep the image defaults here so Dockerfile remains build-focused. Values from
# Docker Compose, `docker run --env`, or an env_file take precedence.
export PATH="/opt/conda/envs/osl/bin:${PATH}"
export CONDA_DEFAULT_ENV="${CONDA_DEFAULT_ENV:-osl}"
export PYTHONPATH="/app${PYTHONPATH:+:${PYTHONPATH}}"
export CC="${CC:-/usr/bin/gcc}"
export CXX="${CXX:-/usr/bin/g++}"
export OSL_SERVER_HOST="${OSL_SERVER_HOST:-0.0.0.0}"
export OSL_SERVER_PORT="${OSL_SERVER_PORT:-8000}"
export OSL_RUNTIME_DIR="${OSL_RUNTIME_DIR:-/app/runtime}"
export OSL_REDIS_URL="${OSL_REDIS_URL:-redis://redis:6379/0}"

PROFILE="${OSL_DOCKER_VQA_DEP_PROFILE:-qwen}"
ROLE="${OSL_DOCKER_SERVICE_ROLE:-api}"

echo "Docker entrypoint starting with OSL_DOCKER_SERVICE_ROLE=${ROLE} OSL_DOCKER_VQA_DEP_PROFILE=${PROFILE}"

case "${PROFILE}" in
  qwen)
    echo "Running opensportslib setup --vqa_qwen"
    opensportslib setup --vqa_qwen
    ;;
  xvars)
    echo "Running opensportslib setup --vqa_xvars"
    opensportslib setup --vqa_xvars
    ;;
  none)
    echo "Skipping opensportslib setup at container startup"
    ;;
  *)
    echo "Unsupported OSL_DOCKER_VQA_DEP_PROFILE=${PROFILE}"
    exit 1
    ;;
esac

# Hardware-specific override is enabled by Dockerfile_spark.
if [[ "${OSL_DOCKER_SPARK:-false}" == "true" ]]; then
  echo "Reinstalling PyTorch with a Blackwell-compatible CUDA build (cu130)"
  python -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cu130 \
    --force-reinstall torch torchvision torchaudio
fi

echo "Launching process: $*"
exec "$@"
