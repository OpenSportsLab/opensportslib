# OpenSportsLib test suite

The suite has one supported entry point:

```bash
bash scripts/run_tests.sh
```

## Fast development suite

The command provisions the default Qwen optional profile (`--pyg --dali
--vqa_qwen`) in its selected Python interpreter, then runs `unit/`, `smoke/`, and
`integration/`. Profile setup is recorded in `setup.log`. These tests cover unit
behavior, public APIs, configuration and data contracts, package architecture,
optional integrations, and bounded synthetic integration workflows. Set
`OSL_TEST_AUTO_SETUP=0` only for a pre-provisioned environment.

Qwen is the default VQA profile. X-VARS has incompatible Transformers pins and
therefore requires a separate environment selected with `OSL_TEST_VQA_PROFILE=xvars`.

Reusable pytest fixtures live in `conftest.py`. Reusable non-fixture builders and
assertions live in `helpers/` only when multiple test modules need them. Stable
declarative fixture files belong in `fixtures/`; generated artifacts use `tmp_path`.

## Heavy release suite

Real-data, GPU, network, and production-model verification lives under `release/`
and is disabled unless explicitly requested. On the prepared release machine, set
`RUN_OSL_RELEASE_TESTS=1` and invoke the same command. The fast gate runs first.

See `tests/release/README.md` for datasets, credentials, caching, optional packages,
and scaling controls. CI branch/tag wiring is intentionally outside this test-suite
contract.

### Full release profiles

Release mode defaults to `OSL_RELEASE_SCALE=full`: complete selected datasets,
canonical preset epoch schedules, fresh checkpoint loading, and the complete
localization E2E matrix. `OSL_RELEASE_SCALE=bounded` is only for a smaller
manual debugging run.

The same runner is invoked on three prepared GPU environments; each produces
`.test-reports/<run-id>/release/release-manifest.json` bound to its Git commit:

```bash
RUN_OSL_RELEASE_TESTS=1 OSL_RELEASE_PROFILE=qwen bash scripts/run_tests.sh
RUN_OSL_RELEASE_TESTS=1 OSL_RELEASE_PROFILE=xvars OSL_TEST_VQA_PROFILE=xvars bash scripts/run_tests.sh
RUN_OSL_RELEASE_TESTS=1 OSL_RELEASE_PROFILE=gar bash scripts/run_tests.sh
```

- `qwen` verifies classification/localization plus published Qwen adapters.
- `xvars` downloads the VideoChatGPT and CLIP dependency models, generates
  X-VARS features/indexes using project tools, then verifies X-VARS training
  and the published adapter.
- `gar` requires approved access to `OpenSportsLab/SoccerNet-GAR@tracking`
  and usable PyG extensions for tracking classification.

Copy the three manifests to the release coordinator and verify that they all
passed for one commit:

```bash
python scripts/verify_release_manifests.py <qwen> <xvars> <gar>
```

Do not download release datasets or models during fast pytest execution.

## Debug reports

Every run writes a timestamped directory under `.test-reports/`, with
`.test-reports/latest` pointing to the newest run. It contains complete logs and
tracebacks, a Markdown diagnostic summary, failed-test IDs, JSON and JUnit reports,
coverage XML, environment details, and a release artifact index when applicable.
Release reports additionally include `release/release-metadata.jsonl`, recording
the selected dataset/revision, materialized config, model family, checkpoint, and
prediction locations without recording credentials.

The fast run enforces `scripts/coverage-baseline.txt`. Raise that number when
sustained coverage improves; never lower it merely to make a change pass.

## Contributor instructions

Human and AI contributors must follow the root `AGENTS.md` and the more specific
`tests/AGENTS.md`. In particular, search and reuse existing fixtures/helpers before
adding support code or creating another test module.
