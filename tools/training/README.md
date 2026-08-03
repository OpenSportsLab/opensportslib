# Training Scripts

Minimal training scripts for each task. Run from the **repository root**.

## Scripts

| Script | Task |
|---|---|
| `classification.py` | Action classification |
| `localization.py` | Action localization |
| `vqa.py` | Visual question answering |

## Arguments

`classification.py` and `localization.py` accept the same CLI arguments:

| Argument | Required | Description |
|---|---|---|
| `--config` | yes | Path to the YAML config file |
| `--train-set` | no | Path to train annotations JSON; defaults to `DATA.train.path` |
| `--valid-set` | no | Path to validation annotations JSON; defaults to `DATA.valid.path` |
| `--test-set` | no | Path to test annotations JSON; defaults to `DATA.test.path` |
| `--weights` | no | Path to pretrained weights |

`vqa.py` accepts the following CLI arguments:

| Argument | Required | Description |
|---|---|---|
| `--config` | yes | Path to the YAML config file |
| `--weights` | no | Optional pretrained weights or adapter path |
| `--resume-from-checkpoint` | no | Hugging Face Trainer checkpoint directory for optimizer/scheduler resume |
| `--train-set` | no | Path to train annotations JSON; defaults to the config train split |
| `--valid-set` | no | Path to validation annotations JSON; defaults to the config valid split |
| `--test-set` | no | Path to test annotations JSON; defaults to the config test split |
| `--skip-infer` | no | Train only and skip the post-training inference/evaluation pass |
| `--use-wandb` | no | Enable Weights & Biases logging |

## Usage

### Classification

```bash
python tools/training/classification.py \
    --config examples/configs/classification_video.yaml
```

With pretrained weights:

```bash
python tools/training/classification.py \
    --config examples/configs/classification_video.yaml \
    --weights OpenSportsLab/OSL-cls-action-mvitv2
```

### Localization

```bash
python tools/training/localization.py \
    --config examples/configs/localization_video_dali.yaml
```

### VQA

```bash
python tools/training/vqa.py \
    --config opensportslib/configs/vqa/qwen.yaml \
    --train-set /path/to/train.json \
    --valid-set /path/to/valid.json \
    --test-set /path/to/test.json
```

Supported VQA options:

- `opensportslib/configs/vqa/xvars.yaml`
- CLIP features + Qwen:
  `opensportslib/configs/vqa/qwen.yaml` for inference,
  `opensportslib/configs/vqa/qwen_lora.yaml` for LoRA training
- `opensportslib/configs/vqa/qwen3_vl_native.yaml`

`qwen3_vl_native.yaml` is the single canonical full end-to-end QwenVL config.
Change `MODEL.components.llm_decoder.params.repo_id` there if you want a
different supported QwenVL checkpoint:

- `Qwen/Qwen3-VL-8B-Instruct`
- `Qwen/Qwen2.5-VL-7B-Instruct`

Train only, without the post-training inference/evaluation pass:

```bash
python tools/training/vqa.py \
    --config opensportslib/configs/vqa/qwen.yaml \
    --train-set /path/to/train.json \
    --valid-set /path/to/valid.json \
    --skip-infer
```

Resume a Hugging Face Trainer run from a saved checkpoint:

```bash
python tools/training/vqa.py \
    --config opensportslib/configs/vqa/qwen.yaml \
    --train-set /path/to/train.json \
    --valid-set /path/to/valid.json \
    --resume-from-checkpoint /path/to/checkpoint-dir
```

## Example Configs

Ready-to-use YAML configs are in `examples/configs/`:

```
examples/configs/classification_video.yaml
examples/configs/classification_sngar_tracking.yaml
examples/configs/localization_video_dali.yaml
opensportslib/configs/vqa/qwen.yaml
opensportslib/configs/vqa/qwen_lora.yaml
opensportslib/configs/vqa/qwen3_vl_native.yaml
opensportslib/configs/vqa/xvars.yaml
```

## Running on Ibex (SLURM)

See [tools/slurm/training/README.md](../slurm/training/README.md) for sbatch job templates.
