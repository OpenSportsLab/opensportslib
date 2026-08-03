# Model Upload Tools

Scripts to upload OpenSportsLib model artifacts to Hugging Face model repos.

This complements the dataset upload tools in `tools/download/` and supports
both:

- folder uploads with `upload_folder(...)`
- single-file uploads with `upload_file(...)`

## Script

- `upload_model_hf.py`
  - Uploads a model folder or a single checkpoint file to a Hugging Face repo.
  - Automatically creates the target repo if it does not exist.
  - Automatically creates the target branch when `--revision` is not `main`.
  - Optionally uploads extra companion files such as `config.yaml` and
    `README.md`.

## Folder upload

Example: upload a validated Qwen3-VL LoRA adapter and ignore training-only
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

Upload the same folder and explicitly attach a config and README:

```bash
python tools/upload/upload_model_hf.py folder \
  --repo-id OpenSportsLab/OSL-VQA-XFOUL-qwen3-8B-VL-lora \
  --folder-path /home/vorajv/opensportslib/checkpoints_vqa_qwen3_vl_native/qwen_vl_native/efc70d9e/qwen_vl_native_lora \
  --config-path /home/vorajv/opensportslib/checkpoints_vqa_qwen3_vl_native/qwen_vl_native/efc70d9e/qwen_vl_native_lora/config.yaml \
  --readme-path /home/vorajv/opensportslib/checkpoints_vqa_qwen3_vl_native/qwen_vl_native/efc70d9e/qwen_vl_native_lora/README.md
```

## File upload

Example: upload a single localization checkpoint file:

```bash
python tools/upload/upload_model_hf.py file \
  --repo-id jeetv/osl-loc-SPL-model \
  --file-path /home/vorajv/opensportslib/checkpoints/train_goal_only/rny002_gsm/2fc8be43/best_checkpoint.pt \
  --path-in-repo model.pt \
  --commit-message "Upload localization checkpoint"
```

Upload the same file and attach a config/model card:

```bash
python tools/upload/upload_model_hf.py file \
  --repo-id jeetv/osl-loc-SPL-model \
  --file-path /home/vorajv/opensportslib/checkpoints/train_goal_only/rny002_gsm/2fc8be43/best_checkpoint.pt \
  --path-in-repo model.pt \
  --config-path /home/vorajv/opensportslib/opensportslib/configs/localization/video_dali.yaml \
  --config-path-in-repo config.yaml \
  --readme-path /home/vorajv/opensportslib/docs/model-zoo.md \
  --readme-path-in-repo README.md
```

## Optional flags

- `--repo-type`
  Defaults to `model`.
- `--revision`
  Upload to a specific branch.
- `--private`
  Create the repo as private if missing.
- `--token`
  Override the local Hugging Face login token.
- `--allow-pattern`
  Repeat to whitelist files during folder upload.
- `--ignore-pattern`
  Repeat to skip files during folder upload.
- `--delete-pattern`
  Repeat to remove matching remote files before folder upload.
- `--extra-file LOCAL=REMOTE`
  Repeat to upload arbitrary companion files after the main upload.

## Notes

- Authenticate first with `huggingface-cli login` unless you pass `--token`.
- For public model repos, prefer excluding training checkpoints, caches, and
  resume artifacts.
- For LoRA repos, upload the adapter folder rather than the entire training run
  directory.
