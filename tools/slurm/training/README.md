# Training SLURM Jobs

Submit training jobs on the Ibex cluster using the provided sbatch scripts.

All commands must be run from the **repository root**.

---

## Scripts

| Script | Task |
|---|---|
| `train_classification.sbatch` | Action classification |
| `train_localization.sbatch` | Action localization |

---

## Default SLURM Parameters

Both scripts share the same defaults, tuned for Ibex:

| Parameter | Value |
|---|---|
| `--partition` | `batch` |
| `--gpus` | `v100:1` |
| `--mem` | `90G` |
| `--time` | `47:59:00` |
| `--cpus-per-task` | `6` |
| `--nodes` | `1` |
| `--ntasks` | `1` |

Logs are written to `ibex_logs/osl_<job_id>.out` and `ibex_logs/osl_<job_id>.err`.

---

## Usage

### Classification

```bash
sbatch tools/slurm/training/train_classification.sbatch
```

By default this runs `examples/quickstart/basic_classification.py`. Edit the script to point to your own training entry point and config:

```bash
# inside train_classification.sbatch
python opensportslib/core/trainer/classification_trainer.py \
    --config /path/to/your/classification_config.yaml
```

### Localization

```bash
sbatch tools/slurm/training/train_localization.sbatch
```

By default this runs `examples/quickstart/basic_localization.py`. Edit the script to point to your own training entry point and config:

```bash
# inside train_localization.sbatch
python opensportslib/core/trainer/localization_trainer.py \
    --config /path/to/your/localization_config.yaml
```

---

## Customizing SLURM Parameters

Override any parameter at submission time without editing the file:

```bash
# Use 2 GPUs and extend the time limit
sbatch --gpus=v100:2 --time=23:59:00 tools/slurm/training/train_classification.sbatch

# Use an account allocation
sbatch --account=conf-neurips-2026.05.15-ghanembs tools/slurm/training/train_localization.sbatch
```

To make changes permanent, edit the `#SBATCH` header lines directly in the script.

---

## Adding a Dataset Download Step

Both scripts include a commented-out download step. Uncomment and adapt it to pre-fetch your dataset before training starts:

```bash
# Classification script
python tools/download/download_osl_hf.py \
    --url https://huggingface.co/datasets/OpenSportsLab/mvfouls/... \
    --dest /ibex/project/c2134/opensportslab/datasets/mvfouls
```

```bash
# Localization script
python tools/download/download_osl_hf.py \
    --url https://huggingface.co/datasets/OpenSportsLab/soccernetpro-localization-snas/... \
    --dest /ibex/project/c2134/opensportslab/datasets/soccernetpro-localization-snas
```

See [tools/download/README.md](../../download/README.md) for full download options.

---

## Monitoring Jobs

```bash
# List your running/pending jobs
squeue -u $USER

# Watch live
watch squeue -u $USER

# Check logs
tail -f ibex_logs/osl_<job_id>.out
tail -f ibex_logs/osl_<job_id>.err
```

---

## Example Configs

Ready-to-use YAML configs are in `examples/configs/`:

```
examples/configs/classification_video.yaml
examples/configs/classification_tracking.yaml
examples/configs/localization.yaml
```

---

## Example training on Ibex

```bash
python tools/training/classification.py \
    --config tools/slurm/training/configs/classification.yaml \
    --train-set /ibex/project/c2134/opensportslab/datasets/soccernetpro-classification-vars/mvfouls/annotations_train.json \
    --valid-set /ibex/project/c2134/opensportslab/datasets/soccernetpro-classification-vars/mvfouls/annotations_valid.json \
    --test-set /ibex/project/c2134/opensportslab/datasets/soccernetpro-classification-vars/mvfouls/annotations_test.json
```
