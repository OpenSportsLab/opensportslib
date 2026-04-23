# Tools

Standalone command-line scripts for converting between the OSL JSON annotation format and a
Parquet + WebDataset representation suited for large-scale ML training pipelines.

## `osl_json_to_parquet_webdataset.py`

```bash
python tools/osl_json_to_parquet_webdataset.py <json_path> <media_root> <output_dir> [options]
```

## `parquet_webdataset_to_osl_json.py`

```bash
python tools/parquet_webdataset_to_osl_json.py <dataset_dir> <output_json_path> [options]
```

## Python API

```python
from opensportslib.tools import convert_json_to_parquet, convert_parquet_to_json
```



## Round-trip example

```bash
# Forward pass: JSON → Parquet + WebDataset
python tools/osl_json_to_parquet_webdataset.py \
    /Users/giancos/git/VideoAnnotationTool/test_data/Localization/gymnastics/annotations.json \
    /Users/giancos/git/VideoAnnotationTool/test_data/Localization/gymnastics \
    /tmp/gymnastics_wds \
    --overwrite

# Backward pass: Parquet + WebDataset → JSON
python tools/parquet_webdataset_to_osl_json.py \
    /tmp/gymnastics_wds \
    /tmp/gymnastics_reconstructed.json



# Description
python tools/osl_json_to_parquet_webdataset.py \
    /Users/giancos/git/VideoAnnotationTool/test_data/Description/xfoul/annotations_test.json \
    /Users/giancos/git/VideoAnnotationTool/test_data/Description/xfoul \
    /Users/giancos/git/VideoAnnotationTool/test_data/xfoul_parquet_webdataset \
    --samples-per-shard 50 \
    --missing-policy skip \
    --overwrite

python tools/parquet_webdataset_to_osl_json.py \
    /Users/giancos/git/VideoAnnotationTool/test_data/xfoul_parquet_webdataset \
    /Users/giancos/git/VideoAnnotationTool/test_data/xfoul_parquet_webdataset_back_to_json/reconstructed_annotations.json \
    --extract-media \
    --output-media-root /Users/giancos/git/VideoAnnotationTool/test_data/xfoul_parquet_webdataset_back_to_json \
    --indent 2


# Classification
python tools/osl_json_to_parquet_webdataset.py \
    /Users/giancos/git/VideoAnnotationTool/test_data/Classification/svfouls/annotations_test.json \
    /Users/giancos/git/VideoAnnotationTool/test_data/Classification/svfouls \
    /Users/giancos/git/VideoAnnotationTool/test_data/svfouls_parquet_webdataset \
    --samples-per-shard 50 \
    --missing-policy skip \
    --overwrite

python tools/parquet_webdataset_to_osl_json.py \
    /Users/giancos/git/VideoAnnotationTool/test_data/svfouls_parquet_webdataset \
    /Users/giancos/git/VideoAnnotationTool/test_data/svfouls_parquet_webdataset_back_to_json/reconstructed_annotations.json \
    --extract-media \
    --output-media-root /Users/giancos/git/VideoAnnotationTool/test_data/svfouls_parquet_webdataset_back_to_json \
    --indent 2
    

# SN-GAR-tracking
python tools/osl_json_to_parquet_webdataset.py \
    /Users/giancos/git/VideoAnnotationTool/test_data/sngar-tracking/annotations_test.json \
    /Users/giancos/git/VideoAnnotationTool/test_data/sngar-tracking \
    /Users/giancos/git/VideoAnnotationTool/test_data/sngar-tracking_parquet_webdataset \
    --samples-per-shard 50 \
    --missing-policy skip \
    --overwrite

python tools/parquet_webdataset_to_osl_json.py \
    /Users/giancos/git/VideoAnnotationTool/test_data/sngar-tracking_parquet_webdataset \
    /Users/giancos/git/VideoAnnotationTool/test_data/sngar-tracking_parquet_webdataset_back_to_json/reconstructed_annotations.json \
    --extract-media \
    --output-media-root /Users/giancos/git/VideoAnnotationTool/test_data/sngar-tracking_parquet_webdataset_back_to_json \
    --indent 2

```




## `download_osl_hf.py`

```bash
python tools/download_osl_hf.py --url <HF_JSON_OR_FOLDER_URL> --output-dir <LOCAL_DIR> --types video --dry-run
```

## `upload_osl_hf.py`

```bash
python tools/upload_osl_hf.py --repo-id <org/repo> --json-path <local_dataset.json> --revision main
```

## Python HF Transfer API

```python
from opensportslib.tools import download_dataset_from_hf, upload_dataset_inputs_from_json_to_hf
```
