# Testing Playbook for AI Coding Agents

This file applies to every file below `tests/` and supplements the repository-root
`AGENTS.md`. Read both files before changing tests. The root file wins if the two
ever conflict.

## Required workflow

1. Identify the production subsystem and its public/configuration/data contracts.
2. Search before writing: use `rg` across `tests/conftest.py`, `tests/helpers/`, and
   the existing tier/subsystem directory for related fixtures, builders, stubs, and tests.
3. Extend the existing owning test module when one exists. Create a new test module
   only for a genuinely distinct subsystem or contract.
4. Reuse existing fixtures and helpers. Add a shared helper only after there are at
   least two real callers; otherwise keep the small helper local to its test module.
5. Cover the successful path and relevant boundary, invalid-input, and regression
   paths. Test public behavior and durable invariants, not private implementation
   details.
6. Run the single supported entry point: `bash scripts/run_tests.sh`.

## Placement rules

| Test kind | Location | Requirements |
| --- | --- | --- |
| Fast unit/API/contract | `tests/unit/<subsystem>/` | Deterministic; no network during test execution |
| Package/API smoke | `tests/smoke/` | Minimal health and initialization checks |
| Bounded workflow integration | `tests/integration/<task>/` | Offline, CPU-compatible, under fast budget |
| Shared pytest fixture | `tests/conftest.py` | Useful across multiple test modules |
| Shared builder/assertion/stub | `tests/helpers/` | Must have at least two callers |
| Stable declarative sample | `tests/fixtures/` | Small, reviewed, and not generated output |
| Real-data/GPU/full-model | `tests/release/` | Explicit opt-in and appropriate markers |

Do not leave collected test modules directly under `tests/`. Place them in exactly
one execution tier and the closest owning subsystem. The runner controls discovery;
do not document or add another pytest entry point.

## Reuse and quality rules

- Do not copy large configuration dictionaries or fixture writers between files.
- Do not create one test file per production source file.
- Do not mock the behavior the test claims to validate. Mock only external systems
  or expensive components outside that test's responsibility.
- Generated videos, checkpoints, predictions, caches, and logs belong in `tmp_path`
  (or the configured release cache), never in committed fixtures.
- Fast tests must not contact Hugging Face, WandB, registries, or other network
  services. The runner may provision its explicit dependency profile before
  pytest starts; that setup output is retained in `setup.log`. Mark genuine
  runtime external tests `network` and place heavy ones in release.
- `infer()` must be checked for in-memory results; persistence is tested separately
  through `save_predictions(...)`.
- When code changes an API, config schema, OSL data format, dependency, packaged
  asset, or task boundary, update the corresponding contract test.
- Intentional breaking changes require updated contract and migration tests,
  migration-facing documentation, and a `BREAKING CHANGE` note.
- Never lower `scripts/coverage-baseline.txt` to make a change pass. Raise it when
  the server report demonstrates a higher sustainable baseline.
- Once release mode is enabled, missing required CUDA, data, credentials, or optional
  training dependencies are failures rather than silent passes.

## Markers

Use the shared markers registered in `tests/conftest.py`: `unit`, `integration`,
`smoke`, `e2e`, `gpu`, `slow`, `release`, `network`, `pretrained`,
`classification`, `localization`, and `vqa`. Do not invent a synonym for an
existing marker.

## Completion checklist

- [ ] Searched for an existing owning test and reusable support code.
- [ ] Added the smallest meaningful test at the correct execution tier.
- [ ] Covered failure messages as well as successful behavior where applicable.
- [ ] Used `tmp_path` for generated artifacts and kept the fast path offline.
- [ ] Updated structural/API/config/data/dependency contracts if affected.
- [ ] Ran `bash scripts/run_tests.sh` and inspected its generated summary.
