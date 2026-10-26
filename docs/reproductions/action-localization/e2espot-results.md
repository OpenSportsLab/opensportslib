# E2E-Spot: results and model releases

**Status: 28 runs queued after T-DEED main results; no new scores or checkpoints yet.**
See the [reproduction plan](e2espot.md) and [action localization catalog](index.md).
The implementation already exists in OpenSportsLib; that alone does not validate
these original-paper experiments. Existing tennis/BAS releases are not relabeled
as new ECCV 2022 reproductions.

## Original-paper targets

Targets below come from **Table 2e** of the
[original ECCV 2022 paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136950033.pdf).
They are published reference numbers, not OpenSportsLib measurements.
The default column reports mAP at ±1 / ±2 native frames from Table 2e. The larger
RGB column reports ±1-frame mAP from Table 3e; no ±2 target is inferred for it.
Preserve each table's marked NMS/dilation settings.

| Dataset / protocol | Paper: default 200MF RGB, ±1 / ±2 | Paper: 800MF RGB, ±1 | OpenSportsLib |
| --- | ---: | ---: | --- |
| Tennis | 96.1 / 97.7 | 96.8 | Not run |
| FS-Comp | 81.0 / 93.5 | 84.0 | Not run |
| FS-Perf | 85.1 / 95.7 | 83.6 | Not run |
| FineDiving | 68.4 / 85.3 | 64.6 | Not run |
| FG-Full | 47.9 / 65.2 | 50.1 | Not run |
| FG-Start | 61.0 / 78.4 | Not reported in Table 3e | Not run |

FG-Full and FG-Start share trained weights. FineGym's default ±1 target is 47.9
in Table 2e/C2 but 47.4 in Table 3. Keep these targets table-specific.

SoccerNet **action** spotting targets from Table 4:

| Model | Test tight average-mAP | Test loose average-mAP | OpenSportsLib |
| --- | ---: | ---: | --- |
| 200MF RGB | 61.19 | 73.25 | Not run |
| 800MF RGB | 61.82 | 74.05 | Not run |

Tight/loose use the original seconds-based SoccerNet protocols, not the precise
frame tolerances above. Challenge targets are recorded separately in
`reproduction/e2espot/targets.json`; private challenge scoring is conditional.
Ablation targets and exact-frame supplementary targets are in the same file.

## SoccerNet Ball Action Spotting extension

Four runs are scheduled in addition to the original-paper experiments: 200MF and
800MF RGB, each with seeds 1/2, on `OpenSportsLab/OSL-SNBAS`. Report dataset
revision, classes and the validated BAS evaluation protocol. This is an extension,
so paper targets and paper-performance gaps are not applicable.

| Model | Seeds | Paper target | OpenSportsLib |
| --- | --- | --- | --- |
| 200MF RGB | 1, 2 | Not applicable | Queued |
| 800MF RGB | 1, 2 | Not applicable | Queued |

## Reporting

For each completed run, show the config/version, dataset revision, seed,
validation-selected epoch and all fixed evaluation protocols. Aggregate seeds
1/2 with mean, standard deviation and signed paper gaps, while retaining each
checkpoint's individual score. These seeds are an OpenSportsLib design choice,
not established author seeds. Mark implementation/preprocessing differences and
scope explicitly; no result is labeled reproduced solely because training ends.

Main results are reported before ablations. The Model Zoo will index validated
releases after they pass fresh-load inference/evaluation checks. Until then,
this page and the planned-entry section of the Model Zoo record pending work.

## Hugging Face release plan

The queued 24 paper runs and four SoccerNet BAS extension runs will share **`OpenSportsLab/OSL-loc-e2e`**.
This is a planned release destination; this campaign has not created or uploaded
models there. Example future revisions:

- `v1-soccernet-small-rgb-seed1`
- `v1-finediving-small-rgb-seed2`
- `v1-fs-comp-big-rgb-seed1`
- `v1-finegym-big-rgb-seed2`
- `v1-tennis-big-rgb-seed1`
- `v1-snbas-small-rgb-seed1`
- `v1-snbas-big-rgb-seed2`
- Later: `v1-fs-perf-small-length-250-seed1`

Single-stream revisions contain only `config.yaml` and `model.pth`, aside from
Hub Git metadata. `main` contains the README, model index, scores and attribution.
Full training/checkpoint/verification evidence remains local. Preserve the
original E2E-Spot BSD-3 notice and appropriate release terms in the README.

Once a revision is actually released, use the shared API:

```python
from opensportslib.apis import LocalizationModel

# Planned example: this revision has not been released by this campaign.
model = LocalizationModel.from_pretrained(
    "OpenSportsLab/OSL-loc-e2e",
    revision="v1-finediving-small-rgb-seed1",
    device="cpu",
)
```

Config and weights must come from the same revision. Pin its commit SHA for an
immutable artifact identity. Author checkpoints, if converted for diagnosis,
remain explicitly labeled as imports and do not count as retrained reproductions.
