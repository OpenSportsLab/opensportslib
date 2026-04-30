# AGENTS.md

This document is the operational playbook for AI coding agents contributing to OpenSportsLib.

## Purpose and Audience

- Audience: AI coding agents (autonomous or semi-autonomous) working in this repository.
- Goal: provide concrete, executable contribution rules for safe and consistent development.

## Library Mission

OpenSportsLib exists to centralize development for AI in sports.

- The library must support multiple modalities (video, tracking, graphs, language, multimodal combinations).
- The library must support multiple tasks through a unified framework.
- Every task must be clearly defined and configured with YAML.

Current task coverage is focused on:

- classification
- localization

Long-term task roadmap includes (non-exhaustive):

- retrieval
- description and dense description
- VQA
- representation learning
- player detection
- tracking
- identification and re-identification
- game state reconstruction
- field localization
- active learning tooling
- test-time adaptation tooling

## Task and Module Architecture Rules

- Each task must be developed as an independent task module/wrapper with its own clear interfaces.
- Task-specific logic should stay in the task module.
- Shared logic should be extracted only when it is truly reused across tasks.
- If a component is shared, implement it as a cross-task module with neutral naming and no hidden coupling to one task.
- Keep the task boundary explicit: task APIs orchestrate, shared modules provide reusable building blocks.

## Source-of-Truth Hierarchy

- `AGENTS.md` defines agent execution workflow and contribution guardrails.
- `CONTRIBUTING.md` and `DEVELOPERS.md` remain the broader policy and architecture references.
- If content overlaps:
  - use `AGENTS.md` for day-to-day agent actions,
  - use `CONTRIBUTING.md` and `DEVELOPERS.md` for additional project context and rationale.

## Environment Bootstrap

Use these commands to set up a working environment:

```bash
git clone https://github.com/OpenSportsLab/opensportslib.git
cd opensportslib

conda create -n osl python=3.12 pip
conda activate osl

pip install -e .

# Install PyTorch (CPU/GPU auto-detected)
opensportslib setup

# Optional dependencies
opensportslib setup --pyg
opensportslib setup --dali
```

## Required Contribution Workflow

1. Start from `dev`:

```bash
git checkout dev
git pull origin dev
```

2. Create a feature/fix branch from `dev`.
3. Open PRs targeting `dev` (not `main`).
4. Use commit message prefixes:
   - `feat:` new feature
   - `fix:` bug fix
   - `refactor:` internal cleanup/refactor
   - `docs:` documentation updates

## Code Change Rules

- Preserve public APIs by default.
- If a breaking API change is unavoidable:
  - include an explicit `BREAKING CHANGE` note in the PR/commit description,
  - update all affected docs and tests in the same change.
- Maintain task-wrapper consistency:
  - shared contract in `BaseTaskModel`,
  - task-specific behavior in task wrappers (`ClassificationModel`, `LocalizationModel`, etc.).
- Keep prediction saving explicit via `save_predictions(...)`; avoid hidden disk-write side effects in `infer()`.

## Testing Contract (Mandatory)

Before opening a PR, run:

```bash
pytest tests/test_*.py
```

When APIs/signatures/behavior change, update the relevant tests, including:

- smoke tests,
- API contract tests,
- integration subset tests.

## Documentation Contract

When API behavior changes, update the relevant docs in the same PR:

- `README.md`
- `opensportslib/apis/README.md`
- `docs/tni/tni.md` and/or `docs/tni/config-guide.md` (when applicable)

## PR Checklist (Agent-Executable)

- [ ] Branch created from `dev`
- [ ] PR targets `dev`
- [ ] Tests pass (`pytest tests/test_*.py`)
- [ ] Documentation updated for behavior/API changes
- [ ] Backward compatibility reviewed
- [ ] Commit/PR description includes API impact summary
- [ ] If breaking: explicit `BREAKING CHANGE` note + migration-facing docs/tests updates

## Related Docs

- `CONTRIBUTING.md`
- `DEVELOPERS.md`
