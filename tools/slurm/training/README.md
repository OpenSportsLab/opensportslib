# Training SLURM Jobs

Submit these Ibex jobs from the repository root. They are specific examples for
the listed datasets; update their paths, account, resources, and Python
arguments for your own experiment.

## Available jobs

| Script | Task and dataset | Entrypoint |
| --- | --- | --- |
| `classification_MVFouls.sbatch` | Classification on SoccerNet MVFouls | `tools/training/classification.py` |
| `classification_XFoul.sbatch` | Classification on OSL-XFoul | `tools/training/classification.py` |
| `localization_SNBAS-2023.sbatch` | Localization on SoccerNet SNBAS 2023 | `tools/training/localization.py` |

All three use the Ibex `batch` partition, one V100 GPU, 90G of memory, six
CPUs, and a 47:59:00 time limit. They activate the `opensportslib` Conda
environment and write `ibex_logs/osl_<job_id>.out` and `.err`.

## Submit a job

```bash
# Classification
sbatch tools/slurm/training/classification_MVFouls.sbatch
sbatch tools/slurm/training/classification_XFoul.sbatch

# Localization
sbatch tools/slurm/training/localization_SNBAS-2023.sbatch
```

Each script supplies its config and train/valid/test manifest paths through the
task wrapper CLI. Those options override the matching canonical config fields:
`DATA.common.splits.train.annotation_path`,
`DATA.common.splits.valid.annotation_path`, and
`DATA.common.splits.test.annotation_path`.

## Customize resources or data

Override scheduler values at submission time:

```bash
sbatch --gpus=v100:2 --time=23:59:00 \
  tools/slurm/training/classification_MVFouls.sbatch
```

For permanent changes, edit the selected script’s `#SBATCH` header and the
`python tools/training/...` invocation. Do not assume the hard-coded Ibex data
paths exist on another cluster. Use the [dataset jobs](../datasets/README.md)
or your own manifests to stage data first.

## Monitor jobs

```bash
squeue -u "$USER"
scontrol show job <job_id>
scancel <job_id>
tail -f ibex_logs/osl_<job_id>.out
```

For local equivalent commands and all CLI options, see
[training scripts](../../training/README.md).
