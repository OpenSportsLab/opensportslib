# Download Tools

Scripts to download and upload OSL datasets via HuggingFace Hub.

## Scripts

- `download_osl_hf.py`
	- Downloads an OSL split by repo, revision, and split name.
	- JSON mode downloads `<split>.json` and all referenced inputs; Parquet mode downloads `<split>/`.
- `download_hf_repo.py`
	- Downloads a full HuggingFace repository snapshot for a given repo and revision.
	- Best when you want the entire repo content for a branch/tag/commit.
- `upload_osl_hf.py`
	- Uploads local dataset inputs from JSON to a HuggingFace dataset repo.
	- Automatically creates the target dataset repo if it does not exist.
	- Automatically creates the target revision branch when `--revision` is not `main` and the branch is missing.

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
	--revision 224p \
	--output-dir /ibex/project/c2134/opensportslab/datasets/OSL-XFoul/224p

# SoccerNet localization SNAS (224p)
python tools/download/download_hf_repo.py \
	--repo-id OpenSportsLab/soccernetpro-localization-snas \
	--revision 224p \
	--output-dir /ibex/project/c2134/opensportslab/datasets/soccernetpro-localization-snas/224p

python tools/download/download_hf_repo.py \
	--repo-id OpenSportsLab/soccernetpro-localization-snas \
	--revision 720p \
	--output-dir /ibex/project/c2134/opensportslab/datasets/soccernetpro-localization-snas/720p

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
for revision in 224p 720p; do
for split in test valid train; do
python tools/download/download_osl_hf.py \
	--repo-id OpenSportsLab/OSL-XFoul --revision $revision --split $split --format parquet \
	--output-dir /ibex/project/c2134/opensportslab/datasets/OSL-XFoul
done
done
```

The split downloader treats `--output-dir` as a root and writes files under
`<output-dir>/<revision>/<split>`.

## Upload

```bash
# JSON mode (upload dataset JSON + referenced input files)
python tools/download/upload_osl_hf.py \
	--repo-id <org/repo> \
	--json-path <local_dataset.json> \
	--split test \
	--format json \
	--revision main

# Parquet mode (convert to Parquet + WebDataset and upload folder)
python tools/download/upload_osl_hf.py \
	--repo-id <org/repo> \
	--json-path <local_dataset.json> \
	--split test \
	--format parquet \
	--shard-size 1GB \
	--revision main
```

```bash
for revision in ResNET_PCA512 224p 720p; do
case "$revision" in
ResNET_PCA512) shard_size="1GB" ;;
224p) shard_size="5GB" ;;
720p) shard_size="20GB" ;;
*) shard_size="5GB" ;;
esac

for split in test valid train challenge; do
python tools/download/upload_osl_hf.py \
	--repo-id OpenSportsLab/OSL-SoccerNet --revision $revision --split $split --format parquet --shard-size $shard_size \
	--json-path /ibex/project/c2134/opensportslab/datasets/soccernetpro-localization-snas/$revision/$split.json
done
done
```



```
## Notes

- Gated repos require accepted access terms and authentication (`huggingface-cli login` or `--token`).
- `download_hf_repo.py` accepts `--repo-type` (`dataset`, `model`, `space`) and optional `--ignore` glob patterns.
- `upload_osl_hf.py` accepts `--format` (`json`, `parquet`).
- In parquet mode, output is uploaded under a folder named after the JSON file stem.

## Python API

```python
from opensportslib.tools import download_dataset_split_from_hf, upload_dataset_inputs_from_json_to_hf
```
