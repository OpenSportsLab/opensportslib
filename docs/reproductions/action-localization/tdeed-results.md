# T-DEED results and model releases

This page records one paper's experiments for [action localization / spotting](index.md).
See the [paper reproduction catalog](../index.md) for the task/algorithm hierarchy.

Main benchmark results are published here and linked from the [Model Zoo](../../model-zoo.md).
The [reproduction guide](tdeed.md) describes training, evaluation and known differences.

## Measured results

Snapshot: 2026-10-25 (UTC). Four completed FineDiving checkpoints are published in
[OpenSportsLab/OSL-loc-TDEED](https://huggingface.co/OpenSportsLab/OSL-loc-TDEED).
Exact paper reproduction remains under review; training completion and artifact
publication do not establish it.

The table reports test mAP in percent, averaged over seeds 1 and 2, using
Soft-NMS. Tolerances are native video frames. Paper targets come from the pinned
extended-paper target file, `reproduction/tdeed/targets.json`.

| Dataset | Model | Completed seeds | Ours ±1 frame | Paper ±1 frame | Ours ±2 frames | Paper ±2 frames |
| --- | --- | --- | --- | --- | --- | --- |
| FineDiving | small | 2 | 71.54 | 71.48 | 89.38 | 87.62 |
| FineDiving | big | 2 | 72.89 | 73.23 | 89.29 | 88.88 |

FineDiving uses the prepared video release; preprocessing, environment and
unpublished author seed choices remain documented differences. These scores are
measured performance, with numerical differences still under review.

SoccerNet action spotting and ball spotting lead the publication queue. Their
scores will be added after evaluation; missing or blocked runs have no score.
SoccerNet action spotting uses tight/loose average mAP; ball spotting uses its
ball protocol. Those metrics will have separate tables from frame-tolerance mAP.
Ablations are deferred until the main-results publication.

## Where to find each artifact

| Artifact | Release location | Contents |
| --- | --- | --- |
| Benchmark comparisons | This page | Paper targets, per-seed scores, mean and standard deviation, gaps, protocol and reproduction status |
| Model discovery | Model Zoo | Validated model releases linked to their configs, cards and measured scores |
| Inference configs and weights | Named revisions of `OpenSportsLab/OSL-loc-TDEED` | `config.yaml` and `model.pth` only |
| Full experiment recipes and evidence | Local campaign and export evidence folders | Resolved training configs, metrics, history, environment, source and dataset hashes |

## Model repository layout

All T-DEED algorithms and trained variants share one Hugging Face model
repository: [OpenSportsLab/OSL-loc-TDEED](https://huggingface.co/OpenSportsLab/OSL-loc-TDEED).
Dataset, benchmark protocol,
architecture, recipe/data version and seed distinguish revisions inside it.
There will be no separate T-DEED model repositories per dataset or model size.

Revision names follow `v<release>-<dataset/protocol>-<variant>-seed<seed>`.
Examples of immutable release tags are:

| Dataset / protocol | Example revisions in `OpenSportsLab/OSL-loc-TDEED` |
| --- | --- |
| SoccerNet action spotting | `v1-soccernet-small-seed1`, `v1-soccernet-big-seed2` |
| SoccerNet ball spotting | `v1-snbas-challenge1-seed1`, `v1-snbas-challenge2-seed2` |
| FigureSkating Competition | `v1-fs-comp-big-seed1`, `v1-fs-comp-small-paper-seed2` |
| FigureSkating Performance | `v1-fs-perf-big-seed1`, `v1-fs-perf-small-paper-seed2` |
| FineDiving | `v1-finediving-small-seed1`, `v1-finediving-big-seed2` |
| FineGym | `v1-finegym-small-seed1`, `v1-finegym-big-seed2` |
| Tennis | `v1-tennis-small-seed1`, `v1-tennis-big-seed2` |
| Later ablations | `v1-finediving-plain-gru-seed1`, `v1-fs-comp-length-200-seed2`, `v1-snbas-challenge1-bas-only-seed1` |

Both seeds will be released for every validated training recipe. FigureSkating
Competition and Performance retain distinct protocol names. FineGym full and
start-event metrics evaluate the same checkpoint. Postprocessing variants reuse
their source weights and identify that dependency in their release metadata.
Ablation revisions will be added after main-results publication.

Each named model revision contains exactly two inference artifacts:
`config.yaml` and `model.pth` (in addition to Hub-managed Git metadata).
`main` contains the repository README, with the model index and aggregate scores.
Choose a revision explicitly and pin its commit SHA when exact immutability is
required. Model weights remain validation-selected; both seeds are retained.

Load directly with the T-DEED-enabled OpenSportsLib development version:

```python
from opensportslib.apis import LocalizationModel

model = LocalizationModel.from_pretrained(
    "OpenSportsLab/OSL-loc-TDEED",
    revision="v1-finediving-small-seed1",
    device="cpu",
)
predictions = model.infer(video_path="diving_clip.mp4", use_wandb=False)
```

The classmethod downloads config and checkpoint together from one cached Hub
snapshot, then loads the weights. `weights_filename` defaults to `model.pth`;
`cache_dir`, `token` and `local_files_only` control downloading. Existing installed
versions without this method require an updated library checkout.

Full optimizer/RNG checkpoints, training configs, histories, predictions,
environment and verification reports remain local. They are not part of the
minimal model revisions. The exporter writes inference artifacts into each
release folder and supporting records into a separate `evidence` directory.

## Publication state

Published FineDiving revisions:

- [v1-finediving-small-seed1](https://huggingface.co/OpenSportsLab/OSL-loc-TDEED/tree/v1-finediving-small-seed1)
- [v1-finediving-small-seed2](https://huggingface.co/OpenSportsLab/OSL-loc-TDEED/tree/v1-finediving-small-seed2)
- [v1-finediving-big-seed1](https://huggingface.co/OpenSportsLab/OSL-loc-TDEED/tree/v1-finediving-big-seed1)
- [v1-finediving-big-seed2](https://huggingface.co/OpenSportsLab/OSL-loc-TDEED/tree/v1-finediving-big-seed2)

Each revision provides only its config and inference checkpoint. Local export
and verification is available through
`python tools/reproduction/export_tdeed.py --output /path/to/new/export`.
This exporter does not upload files.

The implementation remains on the local `feature/t-deed-reproduction` Git branch;
no GitHub push or PR has been performed. Full resume checkpoints and evidence
remain in `outputs/tdeed/campaign` and the local export directories. Main-result
releases precede the ablation campaign.
