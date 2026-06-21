# Training Scripts

Minimal training scripts for each task. Run from the **repository root**.

## Scripts

| Script | Task |
|---|---|
| `classification.py` | Action classification |
| `localization.py` | Action localization |

## Arguments

Both scripts accept the same CLI arguments:

| Argument | Required | Description |
|---|---|---|
| `--config` | yes | Path to the YAML config file |
| `--train-set` | no | Path to train annotations JSON; defaults to `DATA.train.path` |
| `--valid-set` | no | Path to validation annotations JSON; defaults to `DATA.valid.path` |
| `--test-set` | no | Path to test annotations JSON; defaults to `DATA.test.path` |
| `--weights` | no | Path to pretrained weights |

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

## Example Configs

Ready-to-use YAML configs are in `examples/configs/`:

```
examples/configs/classification_video.yaml
examples/configs/classification_sngar_tracking.yaml
examples/configs/localization_video_dali.yaml
```

## Running on Ibex (SLURM)

See [tools/slurm/training/README.md](../slurm/training/README.md) for sbatch job templates.
