# Download Tools

Scripts to download and upload OSL datasets via Hugging Face Hub. These tools
read file references from OSL JSON `data[].inputs[]`; see
`docs/data/osl-json-format.md` for the dataset schema.

## Scripts

- `download_osl_hf.py`
	- Downloads an OSL split by repo, revision, and split name.
	- JSON mode downloads `<split>.json` and all referenced inputs; Parquet mode downloads `<split>/`.
	- `--annotations-only` downloads or reconstructs only `<split>.json`.
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
	--output-dir /ibex/project/c2134/opensportslab/datasets/OSL-XFoul \
	--annotations-only
done
done
```

The split downloader treats `--output-dir` as a root and writes files under
`<output-dir>/<revision>/<split>`.

## Metadata-first and selective downloads

The Python split API accepts `annotations_only=True`. In JSON format it fetches
only `<split>.json`. In Parquet format it fetches only
`<split>/metadata.parquet`, reconstructs `<split>.json`, validates the metadata,
and removes the temporary Parquet file without downloading any WebDataset
shards.

```python
from opensportslib.tools import (
    download_dataset_split_from_hf,
    download_dataset_sample_inputs_from_hf,
)

metadata = download_dataset_split_from_hf(
    repo_id="OpenSportsLab/example",
    revision="main",
    split="test",
    output_dir="downloaded_data",
    download_format="parquet",
    annotations_only=True,
)

assets = download_dataset_sample_inputs_from_hf(
    dataset_json_path=metadata["json_path"],
    sample_id="sample-001",
    # Omit input_path to request every input in the sample.
    input_path="videos/sample-001.mp4",
    overwrite=False,
)
```

Every downloaded JSON records `hf_repo_id`, `hf_branch`, `hf_split`,
`hf_format`, and `hf_commit`. Selective downloads require all five fields so
that later media requests use the same immutable repository commit as the
metadata.

For native JSON repositories, requested assets are downloaded directly. For
Parquet/WebDataset repositories, the required shard is downloaded once to
temporary storage. Requested files obey `overwrite`; every other safely mapped
asset in that shard is extracted only when it is absent locally. Existing
collateral files are never overwritten. Primary inputs and `ball_path`
companions are supported, duplicate paths are coalesced, and temporary shard
artifacts are always removed.

The selective result contains `requested`, `opportunistic`, `overwritten`,
`skipped`, `missing`, and `failed` asset lists. Unsafe destinations—including
absolute paths, traversal outside the JSON directory, symlink escapes, and
directory collisions—are rejected.

Both split download APIs and `download_dataset_sample_inputs_from_hf` accept an
optional `byte_progress_cb(filename, downloaded_bytes, total_bytes)` callback.
It is called while each remote file is transferring, allowing clients to show
file-size progress instead of only item or split counts. For Xet-backed files,
the callback receives Xet's byte increments without replacing the accelerated
transfer with classic HTTP. If Xet is unavailable, explicitly disabled, or not
used by the remote file, the same callback is driven by the HTTP fallback.
Files are written to an atomic temporary path and moved into place only after
completion. A total of `0` means that the remote size is unknown.

## Upload

JSON upload is incremental and tolerant of a partially downloaded local
dataset. It commits the dataset JSON and every referenced input currently
available on disk. Missing references are skipped and reported in
`skipped_missing_input_count` and `skipped_missing_input_paths`; existing
remote files that are not part of the commit are left untouched. Files queued
at an existing repository path update that path.

Parquet + WebDataset upload is strict: every `data[].inputs[].path` and
`ball_path` reference must resolve to a local file before conversion starts.
`find_missing_dataset_inputs(json_path)` provides a preflight list. For a
metadata-first dataset with complete Hugging Face provenance,
`download_dataset_missing_inputs_from_hf(json_path)` hydrates those files from
the recorded immutable commit and rechecks the dataset afterward. It reuses the
selective downloader, so a Parquet shard is downloaded only until its missing
assets have been extracted.

When an existing Parquet split is replaced, the upload commit atomically
deletes obsolete files under the managed `shards/` path while adding the new
metadata, manifest, and shards. Unrelated files in the split folder are left
untouched.

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



## Notes

- Gated repos require accepted access terms and authentication (`huggingface-cli login` or `--token`).
- `download_hf_repo.py` accepts `--repo-type` (`dataset`, `model`, `space`) and optional `--ignore` glob patterns.
- `upload_osl_hf.py` accepts `--format` (`json`, `parquet`).
- In parquet mode, output is uploaded under a folder named after the JSON file stem.
- `annotations_only=True` and `dry_run=True` are mutually exclusive.
- Starting a later full Parquet download completes a metadata-only dataset; an
  existing reconstructed JSON does not cause the shard download to be skipped.

## Python API

```python
from opensportslib.tools import (
    download_dataset_split_from_hf,
    download_dataset_sample_inputs_from_hf,
    upload_dataset_inputs_from_json_to_hf,
)
```
