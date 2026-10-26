# Paper reproductions

OpenSportsLib centralizes implementations and reproducible experiments across
sports AI tasks. This catalog organizes research papers by task, then algorithm.
Each paper has its own implementation guide, experiment scope, measured results
and released models. More algorithms and tasks will extend this structure.

## Papers by task

| Task | Paper / algorithm | Measured scope | Reproduction status | Models |
| --- | --- | --- | --- | --- |
| [Action localization / spotting](action-localization/index.md) | [T-DEED](action-localization/tdeed.md) | FineDiving small/big, two seeds each; other benchmarks in progress | Partial: published test results; exact paper equivalence under review | [OpenSportsLab/OSL-loc-TDEED](https://huggingface.co/OpenSportsLab/OSL-loc-TDEED) |
| [Action localization / spotting](action-localization/index.md) | [E2E-Spot (ECCV 2022)](action-localization/e2espot.md) | 24 paper runs plus 4 SoccerNet BAS extension runs queued | Queued after T-DEED main results; no new scores | `OpenSportsLab/OSL-loc-e2e` (planned) |

Classification, VQA and future task modules can register papers here as their
reproduction records become available. Existing implementations and checkpoints
remain discoverable through the task guides and [Model Zoo](../model-zoo.md);
their presence alone does not establish a paper reproduction.

## What each paper entry contains

- The paper, reference implementation and exact source version.
- Supported task, datasets, splits, annotations and evaluation protocols.
- YAML recipes and commands for preparation, training, inference and evaluation.
- Per-seed results, aggregate scores, paper targets and measured differences.
- A model repository, with revisions for datasets, configurations and seeds.
- Validation evidence, implementation differences and incomplete experiments.

Results carry a scope and status. Entries may be planned, in progress, partially
validated or validated for a stated experiment scope. Released weights or a
completed training job do not by themselves establish numerical equivalence to
the paper. Main results and ablations are tracked separately within each paper.

## Organization

The hierarchy is **task → paper / algorithm → experiments and releases**.
Task wrappers continue to define the public API; an algorithm is one
implementation within its task. A paper's datasets and model variants are
revisions of its OpenSportsLab model repository. The Model Zoo indexes downloads,
while this catalog records research reproduction and benchmark evidence.

For a new paper, add its task entry, an implementation/reproduction guide and a
results page, then link its model releases from the Model Zoo. Reuse the common
task APIs and YAML contracts; record task-specific protocols in that paper's
documentation.
