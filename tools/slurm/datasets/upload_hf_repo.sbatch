#!/bin/bash
#SBATCH --job-name=osl_ul
#SBATCH --output=ibex_logs/osl_ul_%j.out
#SBATCH --error=ibex_logs/osl_ul_%j.err
#SBATCH --partition=batch
#SBATCH --mem=90G
#SBATCH --time=23:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
##SBATCH --account=conf-neurips-2026.05.15-ghanembs

set -euo pipefail

mkdir -p ibex_logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opensportslib

REPO_ID="${1:-}"
REVISION="${2:-}"
JSON_PATH="${3:-}"
UPLOAD_FORMAT="${4:-json}"
SHARD_SIZE="${5:-1GB}"

if [[ -z "${REPO_ID}" || -z "${REVISION}" || -z "${JSON_PATH}" ]]; then
	echo "Usage: sbatch tools/slurm/datasets/upload_hf_repo.sbatch <REPO_ID> <REVISION> <JSON_PATH> [FORMAT] [SHARD_SIZE]"
	echo "Example:"
	echo "  sbatch tools/slurm/datasets/upload_hf_repo.sbatch OpenSportsLab/OSL-SoccerNet 720p-parquet /ibex/project/c2134/opensportslab/datasets/soccernetpro-localization-snas/720p/train.json parquet 20GB"
	exit 1
fi

CMD=(
	python tools/download/upload_osl_hf.py
	--repo-id "${REPO_ID}"
	--json-path "${JSON_PATH}"
	--format "${UPLOAD_FORMAT}"
	--revision "${REVISION}"
)

if [[ "${UPLOAD_FORMAT}" == "parquet" ]]; then
	CMD+=(
		--shard-size "${SHARD_SIZE}"
	)
fi

echo "Running upload command: ${CMD[*]}"
"${CMD[@]}"
