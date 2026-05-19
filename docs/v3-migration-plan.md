# OpenSportsLib V3 Naming Migration Plan (With Temporary V1 Compatibility)

## Goal
Move the codebase to a **single v3 naming model** end-to-end, eliminate duplicate parameters, and keep v1 configs usable during a deprecation window.

## Success Criteria
- All internal runtime/trainer/model/dataset logic reads **one canonical v3 shape**.
- Legacy aliases (`v1` names and compatibility mirror fields) are generated only at controlled boundaries.
- Duplicate settings have explicit precedence and conflict warnings.
- Users can still run existing v1 configs during transition.
- CI enforces no new v1 field usage in core runtime paths.

## Canonical Ownership Model (Decision Complete)
Use this ownership model to remove ambiguity:

- `SYSTEM.*`: process/runtime environment (device selection, gpu count/id, paths, reproducibility)
- `DATA.common/splits/inputs`: data roots, split annotations/source paths, modality sampling and transforms
- `MODEL.*`: architecture/components/topology/load semantics (not execution policy)
- `TRAIN.*`: optimization/training execution behavior

### Critical Rule
`multi_gpu` ownership is **TRAIN execution**, not MODEL:
- Canonical source: `TRAIN.execution.multi_gpu`
- Compatibility mirrors allowed temporarily:
  - read fallback from `MODEL.runtime.multi_gpu` (legacy)
  - emit `MODEL.multi_gpu` only for legacy consumers

## Duplicate Parameter Policy
For each duplicated semantic knob, define precedence and warnings.

### Required precedence rules
1. Canonical v3 field wins.
2. If canonical missing, fallback to legacy alias.
3. If both exist and differ, use canonical and emit warning with both values.
4. If only legacy exists, map forward and emit deprecation warning.

### Initial dedupe targets
- `multi_gpu`: `TRAIN.execution.multi_gpu` vs `MODEL.runtime.multi_gpu` vs legacy `MODEL.multi_gpu`
- `epochs`: `TRAIN.epochs` vs legacy `TRAIN.num_epochs`/`TRAIN.max_epochs`
- split annotation paths: `DATA.common.splits.<split>.annotation_path` vs runtime `DATA.<split>.path` vs `DATA.annotations.<split>`
- split source paths: `DATA.common.splits.<split>.source_path` vs runtime `DATA.<split>.video_path`
- dali/backend flag: `DATA.common.runtime.loader_backend` vs legacy top-level `dali`

## Architecture Changes

### 1) Make canonical config the only internal contract
- Treat `migrate_config(...)` output as the only source for internal business logic.
- Stop allowing business logic to branch directly on v1 field names.
- Keep compatibility adaptation in one place (`core/config/runtime_adapter.py`).

### 2) Boundary-only compatibility layer
- Keep legacy field synthesis in adapter output for old consumers.
- Do not reintroduce legacy fields into canonical state.
- Ensure `resolve_config(...)` always performs:
  1. namespace -> plain dict
  2. migrate to canonical
  3. synchronize runtime overrides into canonical split source-of-truth
  4. adapt to legacy runtime view only if compatibility mode is requested

### 3) Add a config conflict detector
- Implement a single helper in `core/config` that checks duplicate-key conflicts.
- Hook it into load/resolve paths and log warnings with:
  - key name
  - canonical value
  - legacy value
  - effective chosen value
  - deprecation notice

### 4) Introduce compatibility modes
Use one explicit flag in loader API:
- `compatibility="full"` (default for transition): canonical + legacy mirrors
- `compatibility="strict_v3"`: canonical only, no legacy mirrors
- `compatibility="legacy_view"`: current behavior for older consumers

(If changing public signature is risky now, stage this behind internal helper first and expose later.)

## Rollout Phases

### Phase 0: Inventory and lock decisions
- Create a mapping table of all legacy->canonical fields in `core/config` docs.
- Freeze precedence decisions (especially for execution/runtime knobs).

### Phase 1: Normalize load/resolve path
- Ensure all entrypoints (`load_config`, `load_config_omega`, `resolve_config`) produce deterministic canonical state first.
- Keep current override sync fixes (e.g., `valid` -> `valid_data_frames`) canonical-first.

### Phase 2: Dedupe with warnings
- Add conflict detector and deprecation warnings.
- Keep execution behavior unchanged where possible, except where canonical precedence is required.

### Phase 3: Consumer migration
- Update internal consumers progressively to read canonical fields directly.
- Restrict legacy runtime mirror consumption to narrow compatibility shims.

### Phase 4: Strict mode and enforcement
- Add strict-v3 CI target using canonical-only output.
- Fail CI if new code references banned legacy paths in core runtime/trainer/model code.

### Phase 5: Remove redundant mirrors (future)
- After deprecation window, remove legacy mirror writes and fallback reads.

## Backward Compatibility Contract (Temporary)
- v1 config files remain accepted via `migrate_v1_to_v3`.
- Legacy runtime fields remain available in compatibility mode.
- Warnings are non-fatal during transition.
- Release notes document exact deprecation timeline and migration examples.

## Test Strategy

### Unit tests (core/config)
- v1 input -> canonical v3 mapping parity tests.
- duplicate conflict tests for each dedupe target.
- precedence tests: canonical wins, fallback works, warning emitted.
- split override sync tests (including `valid_data_frames` behavior).
- multi_gpu tests for combinations of:
  - `TRAIN.execution.multi_gpu`
  - `MODEL.runtime.multi_gpu`
  - legacy `MODEL.multi_gpu`

### Integration tests
- train/infer smoke for:
  - v1 config path
  - native v3 config path
  - mixed override API path (`train_set`, `valid_set`, `test_set`)
- Verify same effective runtime behavior between equivalent v1 and v3 configs.

### Guardrails in CI
- Lint/grep gate for new direct usage of known legacy keys outside compatibility layer.
- Optional snapshot tests of effective resolved config for representative tasks.

## Implementation Worklist (Concrete)
1. Add `core/config` mapping matrix doc (legacy->canonical + owner + precedence).
2. Add `core/config` conflict detector utility and warning emitter.
3. Wire detector into loader/resolve functions.
4. Refine runtime adapter to treat execution policy fields as TRAIN-owned with documented fallbacks.
5. Add/extend tests for conflict and precedence behavior.
6. Add strict-v3 resolution path and initial CI job (non-blocking first, blocking later).

## Risks and Mitigations
- Risk: silent behavior changes from precedence flips.
  - Mitigation: warnings + parity tests + staged rollout.
- Risk: hidden consumers depend on legacy mirror fields.
  - Mitigation: keep `legacy_view` mode and add telemetry/logging for legacy key access.
- Risk: migration fatigue for external users.
  - Mitigation: publish migration guide with before/after snippets and one-command validation script.

## Deprecation Timeline (Suggested)
- Release N: add warnings + canonical precedence.
- Release N+1: strict-v3 mode documented and recommended.
- Release N+2: strict-v3 default for internal pipelines.
- Release N+3: remove legacy mirrors/fallbacks (or keep behind explicit legacy flag only).
