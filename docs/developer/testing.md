# Developer Testing and Review

OpenSportsLib has one standard command for running its test suite:

```bash
bash scripts/run_tests.sh
```

The command runs the fast smoke, unit, and integration tests and writes reports
under `.test-reports/`. Set `RUN_OSL_RELEASE_TESTS=1` only on a prepared
machine when you need the release suite, which uses real data, GPUs, or network access.

## Put tests in the correct tier

| Change | Test location |
| --- | --- |
| Public API, config schema, builder, metric, or data contract | `tests/unit/` |
| Import/package/CLI health | `tests/smoke/` |
| Offline end-to-end task behavior | `tests/integration/classification`, `localization`, or `vqa` |
| Real models, datasets, network, or GPU | `tests/release/` |

Fast tests must stay deterministic and offline. Use `tmp_path` for generated videos, predictions, checkpoints, and logs. Extend an existing owning test module and reuse fixtures from `tests/conftest.py` before creating helpers.

## Review checklist

1. Confirm the canonical config loads and new fields reach the intended task route.
2. Cover successful, invalid-input, and regression behavior without mocking the behavior under test.
3. Verify `infer()` returns in-memory results and persistence is separately covered through `save_predictions(...)`.
4. Update docs and config/data/API contracts for every public or compatibility change.
5. Run `bash scripts/run_tests.sh`; for docs, run `mkdocs build --strict`.
6. Follow [contributing](../contributing.md) for the `dev` PR target, CLA, and commit-message requirements.
