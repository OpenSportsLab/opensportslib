# Hugging Face Model Upload

OpenSportsLib provides a helper script to upload trained model artifacts to
Hugging Face model repos.

Script:

- `tools/upload/upload_model_hf.py`

This tool supports:

- uploading a full folder with `upload_folder(...)`
- uploading a single file with `upload_file(...)`
- automatically creating the target repo
- automatically creating a non-`main` branch when requested
- optionally uploading companion files such as `config.yaml` and `README.md`

## Folder upload

Example: upload a Qwen3-VL LoRA adapter while ignoring checkpoint and cache
artifacts.

```bash
python tools/upload/upload_model_hf.py folder \
  --repo-id OpenSportsLab/OSL-VQA-XFOUL-qwen3-8B-VL-lora \
  --folder-path /home/vorajv/opensportslib/checkpoints_vqa_qwen3_vl_native/qwen_vl_native/efc70d9e/qwen_vl_native_lora \
  --commit-message "Upload validated Qwen3 8B VL LoRA adapter" \
  --ignore-pattern "checkpoint-*" \
  --ignore-pattern "*resume*" \
  --ignore-pattern "*sanitized*" \
  --ignore-pattern "native_sft_cache/**" \
  --ignore-pattern "*.pyc" \
  --ignore-pattern "__pycache__/**"
```

You can also attach companion files explicitly:

```bash
python tools/upload/upload_model_hf.py folder \
  --repo-id OpenSportsLab/OSL-VQA-XFOUL-qwen3-8B-VL-lora \
  --folder-path /home/vorajv/opensportslib/checkpoints_vqa_qwen3_vl_native/qwen_vl_native/efc70d9e/qwen_vl_native_lora \
  --config-path /home/vorajv/opensportslib/checkpoints_vqa_qwen3_vl_native/qwen_vl_native/efc70d9e/qwen_vl_native_lora/config.yaml \
  --readme-path /home/vorajv/opensportslib/checkpoints_vqa_qwen3_vl_native/qwen_vl_native/efc70d9e/qwen_vl_native_lora/README.md
```

## File upload

Example: upload a single localization checkpoint:

```bash
python tools/upload/upload_model_hf.py file \
  --repo-id jeetv/osl-loc-SPL-model \
  --file-path /home/vorajv/opensportslib/checkpoints/train_goal_only/rny002_gsm/2fc8be43/best_checkpoint.pt \
  --path-in-repo model.pt \
  --commit-message "Upload localization checkpoint"
```

## Useful options

- `--revision`
  Upload to a specific branch.
- `--private`
  Create a private repo if it does not already exist.
- `--allow-pattern`
  Repeat to upload only matching files during folder upload.
- `--ignore-pattern`
  Repeat to skip matching files during folder upload.
- `--delete-pattern`
  Repeat to delete matching remote files before upload.
- `--extra-file LOCAL=REMOTE`
  Repeat to upload arbitrary companion files after the main upload.

## Recommended practice

- Upload validated adapter folders rather than full training directories.
- Exclude `checkpoint-*`, resume artifacts, caches, and sanitized scratch
  outputs from public adapter repos.
- Keep the uploaded `README.md` and `config.yaml` aligned with the checkpoint
  you are publishing.
