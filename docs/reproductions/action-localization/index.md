# Action localization and spotting: paper reproductions

Action localization predicts the temporal positions of actions in sports video.
Action spotting benchmarks evaluate individual event timestamps. Algorithms in
this task use OpenSportsLib's `LocalizationModel` wrapper and YAML configurations;
each paper retains its own data and evaluation protocols.

| Paper / algorithm | Reproduction guide | Results and models | Status |
| --- | --- | --- | --- |
| T-DEED | [Implementation and experiments](tdeed.md) | [Benchmarks and released checkpoints](tdeed-results.md) | FineDiving results released; remaining main benchmarks in progress |
| E2E-Spot (ECCV 2022) | [Original-paper experiment plan](e2espot.md) | [Targets and release plan](e2espot-results.md) | 24 paper runs + 4 BAS extension runs queued after T-DEED main results |

T-DEED and E2E-Spot have independent experiment records for this task. Additional
localization algorithms will be added alongside them.
See [all paper reproductions](../index.md) for other tasks and the shared
documentation structure.
