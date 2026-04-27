# Download Tools

Scripts to download and upload OSL datasets via HuggingFace Hub.

## Scripts

- `download_osl_hf.py`
	- Downloads media referenced by an OSL JSON URL or a folder URL (`tree/...`) from a dataset repo.
	- Best when you want dataset-input files referenced by OSL metadata.
- `download_hf_repo.py`
	- Downloads a full HuggingFace repository snapshot for a given repo and revision.
	- Best when you want the entire repo content for a branch/tag/commit.
- `upload_osl_hf.py`
	- Uploads local dataset inputs from JSON to a HuggingFace dataset repo.

## Full-repo download (recommended for complete branches)

Basic usage:

```bash
python tools/download/download_hf_repo.py \
	--repo-id OpenSportsLab/OSL-XFoul \
	--revision main-parquet \
	--output-dir /ibex/project/c2134/opensportslab/datasets/OSL-XFoul/main-parquet
```

Examples for all repos mentioned so far:

```bash
# OSL-XFoul
python tools/download/download_hf_repo.py \
	--repo-id OpenSportsLab/OSL-XFoul \
	--revision main-parquet \
	--output-dir /ibex/project/c2134/opensportslab/datasets/OSL-XFoul/main-parquet

# SoccerNet localization SNAS (224p)
python tools/download/download_hf_repo.py \
	--repo-id OpenSportsLab/soccernetpro-localization-snas \
	--revision 224p \
	--output-dir /ibex/project/c2134/opensportslab/datasets/soccernetpro-localization-snas/224p

# SoccerNet localization SNAS (ResNET_PCA512)
python tools/download/download_hf_repo.py \
	--repo-id OpenSportsLab/soccernetpro-localization-snas \
	--revision ResNET_PCA512 \
	--output-dir /ibex/project/c2134/opensportslab/datasets/soccernetpro-localization-snas/ResNET_PCA512

# SoccerNet localization SNBAS (224p-2023)
python tools/download/download_hf_repo.py \
	--repo-id OpenSportsLab/soccernetpro-localization-snbas \
	--revision 224p-2023 \
	--output-dir /ibex/project/c2134/opensportslab/datasets/soccernetpro-localization-snbas/224p-2023

# SoccerNet classification VARS (mvfouls)
python tools/download/download_hf_repo.py \
	--repo-id OpenSportsLab/soccernetpro-classification-vars \
	--revision mvfouls \
	--output-dir /ibex/project/c2134/opensportslab/datasets/soccernetpro-classification-vars/mvfouls

# SoccerNet classification GAR (tracking-parquet, gated)
python tools/download/download_hf_repo.py \
	--repo-id OpenSportsLab/soccernetpro-classification-GAR \
	--revision tracking-parquet \
	--output-dir /ibex/project/c2134/opensportslab/datasets/soccernetpro-classification-GAR/tracking-parquet \
	--token hf_xxx

# SoccerNet classification GAR (frames-parquet, gated)
python tools/download/download_hf_repo.py \
	--repo-id OpenSportsLab/soccernetpro-classification-GAR \
	--revision frames-parquet \
	--output-dir /ibex/project/c2134/opensportslab/datasets/soccernetpro-classification-GAR/frames-parquet \
	--token hf_xxx
```

SLURM equivalent using positional args:

```bash
sbatch tools/slurm/datasets/download_hf_repo.sbatch \
	OpenSportsLab/soccernetpro-localization-snas \
	224p \
	/ibex/project/c2134/opensportslab/datasets/soccernetpro-localization-snas/224p
```

## Targeted download from OSL JSON or folder URL

```bash
python tools/download/download_osl_hf.py \
	--url "https://huggingface.co/datasets/OpenSportsLab/OSL-XFoul/tree/main-parquet/test" \
	--output-dir /path/to/data/OSL-XFoul \
	--types all
```

## Upload

```bash
# JSON mode (upload dataset JSON + referenced input files)
python tools/download/upload_osl_hf.py \
	--repo-id <org/repo> \
	--json-path <local_dataset.json> \
	--format json \
	--revision main

# Parquet mode (convert to Parquet + WebDataset and upload folder)
python tools/download/upload_osl_hf.py \
	--repo-id <org/repo> \
	--json-path <local_dataset.json> \
	--format parquet \
	--samples-per-shard 100 \
	--revision main
```

## Notes

- Gated repos require accepted access terms and authentication (`huggingface-cli login` or `--token`).
- `download_hf_repo.py` accepts `--repo-type` (`dataset`, `model`, `space`) and optional `--ignore` glob patterns.
- `upload_osl_hf.py` accepts `--format` (`json`, `parquet`).
- In parquet mode, output is uploaded under a folder named after the JSON file stem.

## Python API

```python
from opensportslib.tools import download_dataset_from_hf, upload_dataset_inputs_from_json_to_hf
```