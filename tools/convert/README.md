# Convert Tools

Scripts to convert between OSL JSON annotations and a Parquet + WebDataset representation.

## Scripts

- `osl_json_to_parquet_webdataset.py`: OSL JSON -> Parquet + WebDataset.
- `parquet_webdataset_to_osl_json.py`: Parquet + WebDataset -> OSL JSON.

## CLI usage

### JSON -> Parquet + WebDataset

```bash
python tools/convert/osl_json_to_parquet_webdataset.py <json_path> <media_root> <output_dir> [options]
```

### Parquet + WebDataset -> JSON

```bash
python tools/convert/parquet_webdataset_to_osl_json.py <dataset_dir> <output_json_path> [options]
```

## Python API

```python
from opensportslib.tools import convert_json_to_parquet, convert_parquet_to_json
```

## Round-trip examples

```bash
# Localization
python tools/convert/osl_json_to_parquet_webdataset.py \
    /path/to/Localization/gymnastics/annotations.json \
    /path/to/Localization/gymnastics \
    /tmp/gymnastics_wds \
    --overwrite

python tools/convert/parquet_webdataset_to_osl_json.py \
    /tmp/gymnastics_wds \
    /tmp/gymnastics_reconstructed.json

# Classification
python tools/convert/osl_json_to_parquet_webdataset.py \
    /path/to/Classification/svfouls/annotations_test.json \
    /path/to/Classification/svfouls \
    /path/to/svfouls_parquet_webdataset \
    --shard-size 500MB \
    --missing-policy skip \
    --overwrite

python tools/convert/parquet_webdataset_to_osl_json.py \
    /path/to/svfouls_parquet_webdataset \
    /path/to/svfouls_back_to_json/reconstructed_annotations.json \
    --extract-media \
    --output-media-root /path/to/svfouls_back_to_json \
    --indent 2

# SN-GAR-tracking
python tools/convert/osl_json_to_parquet_webdataset.py \
    /path/to/sngar-tracking/annotations_test.json \
    /path/to/sngar-tracking \
    /path/to/sngar-tracking_parquet_webdataset \
    --shard-size 500MB \
    --missing-policy skip \
    --overwrite

python tools/convert/parquet_webdataset_to_osl_json.py \
    /path/to/sngar-tracking_parquet_webdataset \
    /path/to/sngar-tracking_back_to_json/reconstructed_annotations.json \
    --extract-media \
    --output-media-root /path/to/sngar-tracking_back_to_json \
    --indent 2
```
