# Developer Testing and Review

OpenSportsLib has one supported test entry point:

```bash
bash scripts/run_tests.sh
```

The command runs the fast smoke, unit, and integration suites and writes diagnostics under `.test-reports/`. Set `RUN_OSL_RELEASE_TESTS=1` only on a prepared machine for the real-data/GPU/network release suite.

## Coverage and reports

The fast run measures statements and branches for the library and server
separately. The minimums are 50% for `opensportslib/` and 20% for `server/`,
controlled by `scripts/coverage-baseline.txt` and
`scripts/server-coverage-baseline.txt`. Each gate must pass independently.

Inspect `coverage-opensportslib.txt` and `coverage-server.txt` under
`.test-reports/latest/` for missing lines and branch paths. Matching XML reports
and the combined `coverage.xml` are available for tooling. GitHub Actions retains
the test summaries, JSON/JUnit results, and coverage reports for 14 days, even
when tests fail. Coverage data stays inside the run's report directory.

Prioritize tests of observable behavior: data boundaries, event matching,
checkpoint restoration, registration state changes, and inference failures.
Use small CPU models and mock only external boundaries. A higher coverage
percentage does not replace real-data/GPU release verification.

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
