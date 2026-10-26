# E2E-Spot: reproduction plan

**Task:** action localization / spotting. **Status:** queued after T-DEED main results; existing algorithm,
new paper-specific reproduction campaign. See [results and release targets](e2espot-results.md)
and the [action localization catalog](index.md).

The reference paper is Hong et al., *Spotting Temporally Precise, Fine-Grained
Events in Video*, ECCV 2022. Sources:
[paper](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136950033.pdf),
[supplement](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136950033-supp.pdf),
[author code](https://github.com/jhong93/spot), and
[released model configs](https://github.com/jhong93/e2e-spot-models).

The code reference is pinned to `edec4201471beed631bed374bd0b95fcdc8a2f4f`;
the model-config repository to `10d07a4fb01f2eb76785e15552e5648bf0793509`.
The machine-readable plan is `reproduction/e2espot/matrix.json`, with paper
scores in `targets.json`, 13 pinned author RGB JSON configs in `reference_configs`,
and their hashes/provenance in `reference-models.json`.
Fourteen canonical OpenSportsLib YAML recipes are in
`opensportslib/configs/localization/e2espot/`; these cover the paper runs and BAS
extension. Recipe/data/model verification remains a prerequisite to execution.

## Scope and order

The persistent queue is defined in `reproduction/main-queue.yaml`. It schedules
**24 original-paper runs plus 4 SoccerNet Ball Action Spotting runs**, all with
seeds 1 and 2, after every T-DEED main-result seed is completed. It checks actual
result files rather than treating a campaign exit, blocked dataset or failure as
completion. E2E-Spot ablations are not scheduled.

| Phase | Experiments | Training runs |
| --- | --- | ---: |
| Main A | Default 200MF RGB on five precise-spotting protocols; SoccerNet 200MF/800MF RGB | 14 |
| Main B | 800MF RGB on those five protocols, matching Table 3e | 10 |
| BAS extension | SoccerNet Ball Action Spotting, 200MF and 800MF RGB | 4 |
| Later ablations | Table 3 architecture/context/resolution studies; outside current queue | 148 |

The five precise-spotting protocols are Tennis, FS-Comp, FS-Perf, FineDiving and
FineGym. FG-Full and FG-Start evaluate the same FineGym weights. Main A/B contain
12 recipes / 24 seeded runs; adding the BAS extension gives 14 recipes / 28 runs.
The larger RGB rows reuse Main B outputs during ablation reporting. Independent
OpenSportsLib seeds 1/2 are a design choice, not identified author seeds.

### Main A: RGB models

| Dataset/protocol | Architecture | Training cycles | Source config prefix |
| --- | --- | --- | --- |
| SoccerNet Ball Action Spotting | `OpenSportsLab/OSL-SNBAS` | Validate the prepared annotation revision, label ordering, 25-FPS media and official BAS metric |
| SoccerNet action spotting | RegNet-Y 200MF + GSM + GRU | 100 | `soccer_rny002gsm_gru_rgb` |
| SoccerNet Ball Action Spotting | `OpenSportsLab/OSL-SNBAS` | Validate the prepared annotation revision, label ordering, 25-FPS media and official BAS metric |
| SoccerNet action spotting | RegNet-Y 800MF + GSM + GRU | 150 | `soccer_rny008gsm_gru_rgb` |
| Tennis | RegNet-Y 200MF + GSM + GRU | 50 | `tennis_rny002gsm_gru_rgb` |
| FS-Comp | RegNet-Y 200MF + GSM + GRU | 50 | `fs_comp_rny002gsm_gru_rgb` |
| FS-Perf | RegNet-Y 200MF + GSM + GRU | 50 | `fs_perf_rny002gsm_gru_rgb` |
| FineDiving | RegNet-Y 200MF + GSM + GRU | 50 | `finediving_rny002gsm_gru_rgb` |
| FineGym | RegNet-Y 200MF + GSM + GRU | 100 | `finegym_rny002gsm_gru_rgb` |

These are original E2E-Spot recipes, not the newer T-DEED settings or the existing
SpoTTA configuration. The latter's adaptation must be disabled for this paper.

### Main B: larger RGB models

For each of Tennis, FS-Comp, FS-Perf, FineDiving and FineGym, train the
released 800MF RGB recipe (50 cycles, or 150 for FineGym). These reproduce the
scale comparison in Table 3e.

### SoccerNet Ball Action Spotting extension

Two standalone E2E-Spot models, RegNet-Y 200MF and 800MF with GSM and GRU, each
use seeds 1/2 on `OpenSportsLab/OSL-SNBAS`. These are four additional runs, not
replacements for the 24 paper runs. They are explicitly labeled extensions:
BAS was not reported in the original ECCV 2022 paper, so there is no paper target.

The initial extension recipe uses 50 cycles, learning rate 0.001, 500K frames per
cycle, full-rate 25-FPS sampling, batch 8 and no crop. Class order comes from the
prepared SNBAS manifests. The BAS-specific evaluation/timestamp/label-version
checks must pass before execution; do not reuse the SoccerNet action-spotting
seconds protocol. Small and big configs are separate canonical YAMLs.

### Later: Table 3 ablations

Every row below applies to all five precise-spotting protocols unless noted.
All alter the default 200MF RGB recipe, except the larger-backbone comparison already covered by Main B.

| Group | Variants | New training recipes |
| --- | --- | ---: |
| Temporal context, Table 3a | Clip lengths 8, 16, 25, 50, 250, 500; no FineDiving-500 target was reported | 29 |
| Temporal components, Table 3b | No GRU; TSM instead of GSM; no GSM; neither GSM nor GRU | 20 |
| Spatial resolution, Table 3c | 112-pixel height/crop | 5 |
| Temporal heads, Table 3d | MS-TCN, ASFormer, 3-layer GRU, multiscale GRU* | 20 |

The 100-frame controls and larger RGB rows reuse main runs. The later ablations
total 74 recipes / 148 seeded training runs. Ablation recipes
need reconciliation with the paper and source before runnable YAMLs are enabled;
changing one parameter is a controlled experiment until parity is established.
GRU* needs an implementation audit/addition: the current head builder has GRU,
deeper GRU, MS-TCN and ASFormer, but no explicit GRU* branch.

### Evaluation-only and extended comparisons

- Supplement C4/C5 default RGB rows: score saved outputs at frame tolerances 0, 1 and 2, with no
  NMS and windows 1, 2 and 4. Preserve separate protocol scores; do not tune a
  published default on test labels. Report the paper's raw/NMS selections as
  explicitly identified comparisons. FG-Start uses the original 16-class subset.
- Per-class precision/recall: reuse predictions; compare the default model's
  class behavior, as in supplemental Figure C1.
- Efficiency, Table 1: profile matching models, clip/batch sizes, FLOPs, memory
  and throughput. Record hardware/software; GB10 timing is not A5000 replication.
- SoccerNet challenge, Table 4: evaluate the two standard models only through the
  appropriate held-out protocol. The additional 800MF train+val+test model is a
  separate recipe, for challenge evaluation only. Private labels/evaluation access
  are prerequisites; never score it on its own training test split as a held-out result.
- Full comparison coverage, Table 2a–d and supplement C1–C3: inventory the exact
  source-reported RGB-only combinations of MViT-B, TSP and (K+1)-VC features with
  temporal heads and applicable CALF/dilation settings. Include VC-Spot separately.
  This is an additional feature-extraction/baseline workstream, not part of the
  24-run E2E-Spot main-model budget. Unreported Cartesian-product combinations
  are not reproduction experiments.

## Dataset preparation and reuse

`reproduction/e2espot/data-compatibility.json` compares the pinned E2E-Spot and
T-DEED released annotations. Class order, split membership, event labels/native
frames and reference frame counts match for all five precise-spotting protocols.
This permits shared prepared data after media checks; it does not verify pixels,
augmentation, model preprocessing or source-video availability.

| Dataset | Preferred existing source | Preparation / validation needed |
| --- | --- | --- |
| Tennis | `OpenSportsLab/OSL-loc-tennis-public`, `swing-data` | Current 144p clips are below original 398×224 preparation; acquire suitable source resolution and preserve released native frame axes |
| FigureSkating | `OpenSportsLab/OSL-loc-FigureSkating` | Validate both split protocols and full-broadcast performance offsets; retain source-encoding substitutions in provenance |
| FineDiving | `OpenSportsLab/OSL-loc-FineDiving` | Preserve source frame geometry before 224 crops; do not reuse T-DEED's square-resize preprocessing implicitly |
| FineGym | `OpenSportsLab/OSL-loc-gymnastics` | Require all 5,374 padded performances and exact timing; inspect/fix the existing source-FPS rebuild failure before readiness |
| SoccerNet Ball Action Spotting | `OpenSportsLab/OSL-SNBAS` | Validate the prepared annotation revision, label ordering, 25-FPS media and official BAS metric |
| SoccerNet action spotting | `SoccerNet/SoccerNet_raw_HQ` | Prefer the official `videos-224p` / `videos-720p` branches; audit decoded media, original extraction stride, timestamps and `fps.txt` behavior; do not silently substitute nominal 2 FPS for actual sampled clocks |

The paper's prose counts differ from its released annotations for Tennis and
FineDiving. Record this discrepancy rather than treating prose counts as a reason
to alter the released splits. Pin annotation/media revisions and checksums.

## Queue operation

The supervisor can be inspected or resumed with:

```bash
python tools/reproduction/paper_queue.py status
python tools/reproduction/paper_queue.py preflight
python tools/reproduction/paper_queue.py watch
```

The watcher records `outputs/reproduction/main-queue/status.json`, holds a
single-worker lock and checks the predecessor every 60 seconds. It does not
interrupt T-DEED or enable either paper's ablations. Once T-DEED's 32 configured
main seeds have valid result files, it selects ready E2E-Spot jobs in matrix order
and runs one at a time. Completed E2E-Spot seeds are skipped; failures are recorded
in their own `failure.json` and are not retried automatically.

Execution also requires data readiness and per-recipe verification in
`outputs/e2espot/verification.json`. Each record under `experiments.<id>` needs
`status: passed`, current `config_sha256`, `implementation_sha256` and
`manifest_sha256`, plus successful `model_parity`, `dataset_alignment`,
`metric_parity` and `repeatability` checks. The preflight command reports the
expected identities and missing prerequisites. These are evidence from actual
validation runs, not manual approval flags. Changed configs, code or manifests
invalidate that evidence. Until these checks are available, jobs stay visibly
blocked rather than producing an unvalidated reproduction result.

## Validation required before execution

The existing algorithm is in `opensportslib/models/base/e2e.py`, with task
orchestration in `LocalizationModel` and `localization_trainer.py`. Reuse these
interfaces; a new paper entry does not justify a new public task wrapper.

1. Import an author checkpoint as a diagnostic and verify state-key mapping,
   backbone/GSM behavior, GRU dimensions, preprocessing and logits against the
   pinned source. Keep imported-author and independently trained provenance distinct.
2. Audit the generated canonical YAMLs against the pinned main references. Preserve 500K-frame cycles,
   batch 8, 100-frame default windows, AdamW learning rate 0.001, foreground weight
   5, chained warmup/cosine schedule and released validation start cycles. Match
   FigureSkating label dilation and no-crop SoccerNet settings.
3. Implement/audit an E2E-compatible native-frame dataset adapter for the prepared
   manifests, performance offsets, padded clips, random-window sampling and mixup.
   Current generic broadcast defaults are not proof of paper-specific data parity.
4. Compare precise AP, overlap averaging, prediction thresholds, NMS and FG-Start
   scoring with the original evaluator. Audit SoccerNet's half-tolerance convention
   and official tight/loose protocols separately. Disable SpoTTA throughout.
5. Add seed/RNG provenance, reliable resume/selection and paper-specific run/score
   ledgers. The existing E2E checkpoint saver does not record the full Python,
   NumPy, CPU and CUDA RNG states used by the T-DEED campaign.
6. Export checkpoints with matching configs and verify revision-aware loading.
7. Verify repeatability and fresh model loading, then write the validation records.
   Store all seed scores, means/std, signed paper gaps and remaining differences.

The queue is armed; no E2E-Spot training begins ahead of its predecessor and
validation requirements. No GitHub push or new model upload is part of scheduling.
Results and model releases follow the same validation and publication gates.
