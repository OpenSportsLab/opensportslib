#!/bin/bash
#SBATCH --job-name=osl_dl
#SBATCH --output=ibex_logs/osl_dl_%j.out
#SBATCH --error=ibex_logs/osl_dl_%j.err
#SBATCH --partition=batch
#SBATCH --mem=90G
#SBATCH --time=23:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
##SBATCH --account=conf-neurips-2026.05.15-ghanembs

# ---------------------------------------------------------------------------
# Parameterized full-repo HuggingFace download
#
# Usage:
#   sbatch tools/slurm/datasets/download_hf_repo.sbatch <REPO_ID> [REVISION] [OUTPUT_DIR] [HF_TOKEN]
#
# Arguments:
#   $1  REPO_ID    — HuggingFace repo ID, e.g. OpenSportsLab/OSL-XFoul       (required)
#   $2  REVISION   — branch / tag / commit SHA                                 (default: main)
#   $3  OUTPUT_DIR — local destination directory                               (default: /ibex/project/c2134/opensportslab/datasets/<repo_name>)
#   $4  HF_TOKEN   — access token for gated repos                              (default: unset)
# ---------------------------------------------------------------------------

set -euo pipefail

mkdir -p ibex_logs

# --- Parse positional arguments --------------------------------------------
REPO_ID="${1:-}"
REVISION="${2:-main}"

if [[ -z "${REPO_ID}" ]]; then
    echo "ERROR: REPO_ID is required as the first argument."
    echo "Usage: sbatch $0 <REPO_ID> [REVISION] [OUTPUT_DIR] [HF_TOKEN]"
    exit 1
fi

REPO_NAME="${REPO_ID##*/}"
OUTPUT_DIR="${3:-/ibex/project/c2134/opensportslab/datasets/${REPO_NAME}}"
HF_TOKEN="${4:-}"
REPO_TYPE="dataset"

echo "=========================================="
echo "Repo     : ${REPO_ID}"
echo "Version  : ${REVISION}"
echo "Type     : ${REPO_TYPE}"
echo "Dest     : ${OUTPUT_DIR}"
echo "=========================================="

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opensportslib

mkdir -p "${OUTPUT_DIR}"
pwd

EXTRA_ARGS=""
if [[ -n "${HF_TOKEN}" ]]; then
    EXTRA_ARGS="--token ${HF_TOKEN}"
fi

python tools/download/download_hf_repo.py \
    --repo-id   "${REPO_ID}" \
    --revision  "${REVISION}" \
    --repo-type "${REPO_TYPE}" \
    --output-dir "${OUTPUT_DIR}" \
    ${EXTRA_ARGS}
