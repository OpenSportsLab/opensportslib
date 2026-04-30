# Training Scripts

Minimal training scripts for each task. Run from the **repository root**.

## Scripts

| Script | Task |
|---|---|
| `basic_classification.py` | Action classification |
| `basic_localization.py` | Action localization |

## Arguments

Both scripts accept the same CLI arguments:

| Argument | Required | Description |
|---|---|---|
| `--config` | yes | Path to the YAML config file |
| `--train-set` | yes | Path to train annotations JSON |
| `--valid-set` | yes | Path to validation annotations JSON |
| `--test-set` | yes | Path to test annotations JSON |
| `--weights` | no | Path to pretrained weights |

## Usage

### Classification

```bash
python tools/training/basic_classification.py \
    --config examples/configs/classification_video.yaml \
    --train-set /path/to/train_annotations.json \
    --valid-set /path/to/valid_annotations.json \
    --test-set /path/to/test_annotations.json
```

With pretrained weights:

```bash
python tools/training/basic_classification.py \
    --config examples/configs/classification_video.yaml \
    --weights /path/to/weights.pt \
    --train-set /path/to/train_annotations.json \
    --valid-set /path/to/valid_annotations.json \
    --test-set /path/to/test_annotations.json
```

### Localization

```bash
python tools/training/basic_localization.py \
    --config examples/configs/localization.yaml \
    --train-set /path/to/train_annotations.json \
    --valid-set /path/to/valid_annotations.json \
    --test-set /path/to/test_annotations.json
```

## Example Configs

Ready-to-use YAML configs are in `examples/configs/`:

```
examples/configs/classification_video.yaml
examples/configs/classification_tracking.yaml
examples/configs/localization.yaml
```

## Running on Ibex (SLURM)

See [tools/slurm/training/README.md](../slurm/training/README.md) for sbatch job templates.
