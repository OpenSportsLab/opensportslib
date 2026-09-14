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
export OSL_CLASSIFICATION_CONFIG_PATH="${OSL_CLASSIFICATION_CONFIG_PATH:-/app/config/classification_video.standalone.yaml}"
export OSL_LOCALIZATION_CONFIG_PATH="${OSL_LOCALIZATION_CONFIG_PATH:-/app/config/localization_video_dali.standalone.yaml}"
export OSL_VQA_QWEN3_CONFIG_PATH="${OSL_VQA_QWEN3_CONFIG_PATH:-/app/config/vqa_qwen3_vl_native.standalone.yaml}"
export OSL_VQA_QWEN25_CONFIG_PATH="${OSL_VQA_QWEN25_CONFIG_PATH:-/app/config/vqa_qwen2_5_vl_native.standalone.yaml}"
export OSL_VQA_XVARS_CONFIG_PATH="${OSL_VQA_XVARS_CONFIG_PATH:-/app/config/vqa_xvars.standalone.yaml}"

PROFILE="${OSL_DOCKER_VQA_DEP_PROFILE:-qwen}"
ROLE="${OSL_DOCKER_SERVICE_ROLE:-api}"
PREDOWNLOAD_ON_START="${OSL_DOCKER_PREDOWNLOAD_ON_START:-false}"
PREDOWNLOAD_WORKER_ONLY="${OSL_DOCKER_PREDOWNLOAD_WORKER_ONLY:-true}"

echo "Docker entrypoint starting with OSL_DOCKER_SERVICE_ROLE=${ROLE} OSL_DOCKER_VQA_DEP_PROFILE=${PROFILE}"

should_predownload="false"
if [[ "${PREDOWNLOAD_ON_START}" == "true" ]]; then
  if [[ "${PREDOWNLOAD_WORKER_ONLY}" == "true" ]]; then
    if [[ "${ROLE}" == "worker" ]]; then
      should_predownload="true"
    else
      echo "Skipping Hugging Face predownload on non-worker role ${ROLE}"
    fi
  else
    should_predownload="true"
  fi
fi

if [[ "${should_predownload}" == "true" ]]; then
  echo "Predownloading configured Hugging Face assets before service startup"
  /app/scripts/download_all_weights.sh
else
  echo "Skipping Hugging Face predownload at container startup"
fi

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
