# Download Tools

Scripts to download and upload OSL dataset files via HuggingFace Hub.

## Scripts

- `download_osl_hf.py`: download dataset inputs from a HuggingFace JSON/folder URL.
- `upload_osl_hf.py`: upload local dataset inputs from JSON to a HuggingFace dataset repo.

## Common dataset URLs

This helper downloads train/valid/test split folders from these repositories:

- `OpenSportsLab/OSL-XFoul` at revision `main-parquet`
- `OpenSportsLab/soccernetpro-classification-GAR` at revision `tracking-parquet`
- `OpenSportsLab/soccernetpro-classification-GAR` at revision `frames-parquet`

Notes:

- `soccernetpro-classification-GAR` branches are gated datasets. You must accept access conditions on HuggingFace and authenticate (`huggingface-cli login` or `--token`).

Examples:

```bash
# OSL-XFoul (test set only)
python tools/download/download_osl_hf.py \
	--url "https://huggingface.co/datasets/OpenSportsLab/OSL-XFoul/tree/main-parquet/test" \
	--output-dir /path/to/data/OSL-XFoul \
	--types all

# GAR tracking-parquet (test set only - gated)
python tools/download/download_osl_hf.py \
	--url "https://huggingface.co/datasets/OpenSportsLab/soccernetpro-classification-GAR/tree/tracking-parque/test" \
	--output-dir /path/to/data/soccernetpro-classification-GAR-tracking \
	--types all

# GAR frames-parquet (test set only - gated)
python tools/download/download_osl_hf.py \
	--url "https://huggingface.co/datasets/OpenSportsLab/soccernetpro-classification-GAR/tree/frames-parquet/test " \
	--output-dir /path/to/data/soccernetpro-classification-GAR-frames \
	--types all
```

## CLI usage

### Download

```bash
python tools/download/download_osl_hf.py --url <HF_JSON_OR_FOLDER_URL> --output-dir <LOCAL_DIR> --types video --dry-run
```

### Upload

```bash
python tools/download/upload_osl_hf.py --repo-id <org/repo> --json-path <local_dataset.json> --revision main
```

## Python API

```python
from opensportslib.tools import download_dataset_from_hf, upload_dataset_inputs_from_json_to_hf
```