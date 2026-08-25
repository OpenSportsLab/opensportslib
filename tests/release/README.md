# Release verification tests

Extensive, real-data, real-training tests for confirming that training still
works end-to-end for (most of) the model families OpenSportsLib ships, after
a big release. **These are not part of the regular test contract** — they are
not run by `pytest tests/test_*.py` (AGENTS.md's mandatory pre-PR command,
which only globs flat files directly under `tests/`) and every test here
additionally skips itself unless `RUN_OSL_RELEASE_TESTS=1` is set, so they
never run by accident even under a broad `pytest tests/`.

Do not add these to CI. Run them manually, on a machine with a real GPU, real
disk space, and time to spare.

## What's covered

Every dataset fixture prefers a dataset pinned in the
[`OpenSportsLab/osl-ready-datasets`](https://huggingface.co/collections/OpenSportsLab/osl-ready-datasets)
Hugging Face collection — the org's sharded (parquet + webdataset) datasets —
and falls back to a known-good, real, but loose-file dataset only while the
sharded one isn't populated yet. See "Preferring OSL-ready (sharded)
datasets" below.

| File | Task | OSL-ready (preferred) | Fallback (used until the above is published) | Model families |
| --- | --- | --- | --- | --- |
| `test_classification_release.py` | classification | `OpenSportsLab/OSL-XFoul`¹ | `OpenSportsLab/OSL-cls-UEFA-fouls` (gated) | MVNetwork (r3d_18, mc3_18, r2plus1d_18, s3d, mvit_v2_s), HF VideoMAE full-model |
| `test_localization_release.py` | localization (E2E) | `OpenSportsLab/OSL-SNBAS` | `OpenSportsLab/soccernetpro-localization-tennis` (public) | E2E (rn18/rn50/rny002/rny008_gsm/convnextt x gru/deeper_gru/mstcn/asformer), E2E+DALI |
| `test_localization_release.py` | localization (features) | `OpenSportsLab/OSL-SoccerNet` | `OpenSportsLab/SoccerNet-ActionSpotting-Features` (public, ~111GB full) | ContextAware (CALF), LearnablePooling (NetVLAD++) |
| `test_vqa_release.py` | VQA | `OpenSportsLab/OSL-XFoul`¹ | none — skips if unpublished | X-VARS/VideoChatGPT LoRA, CLIP+Qwen LoRA, native QwenVL LoRA |

¹ OSL-XFoul is dual-purpose: the same RefPal/MVFouls-style clips carry both
classification labels and VQA question/answers, so it's the preferred
dataset for both tasks.

As of the last check, all three OSL-ready datasets (`OSL-XFoul`, `OSL-SNBAS`,
`OSL-SoccerNet`) are unpublished placeholder repos, so every fixture
currently takes its fallback path. Re-run after any of them lands to switch
automatically — no code change needed.

Known gaps, marked `skip` with a reason rather than a fake pass:

- The `frames_npy` VideoModel classification family (dinov3, clip, videomae,
  videomae2 as pure feature extractors) needs a raw-video → frame-`.npy`
  pre-extraction step this suite doesn't implement yet.
- Tracking-modality classification (`graph_conv`,
  `classification/sngar_tracking.yaml`) depends on
  `OpenSportsLab/SoccerNet-GAR`, which is an unpublished placeholder repo as
  of this writing.
- Retrieval and description/captioning have no first-class training workflow
  in the library yet (per the README's roadmap section), so there's nothing
  to verify here.

## Prerequisites

- A working `opensportslib` install with GPU support (`opensportslib setup`).
- For the VQA backends: `opensportslib setup --vqa_xvars` and/or
  `opensportslib setup --vqa_qwen`. Tests skip cleanly (not fail) if the
  relevant optional dependency isn't installed.
- For DALI E2E localization: `opensportslib setup --dali`. Skips cleanly if
  absent.
- `HF_TOKEN` (or `HUGGINGFACE_TOKEN`) exported for any gated dataset —
  currently `OSL-cls-UEFA-fouls`. Request access on the dataset's HF page
  first; tests skip with instructions if access isn't granted.
- Disk space: the tennis and UEFA-fouls datasets are small (~250MB–1.4GB),
  but the full `SoccerNet-ActionSpotting-Features` dataset is ~111GB and the
  QwenVL-native VQA backend downloads an 8B-parameter model. Set
  `OSL_RELEASE_DATA_DIR` to point dataset downloads at a disk with room —
  and reuse it across runs, since already-downloaded files won't be
  re-fetched.

## Running

```bash
# Everything (expensive -- real training runs across every backbone/head
# combination). Dataset downloads are capped by default (see "Why file
# counts are capped" below), so this is real but bounded, not 100GB+:
RUN_OSL_RELEASE_TESTS=1 pytest tests/release -v -s
RUN_OSL_RELEASE_TESTS=1 OSL_RELEASE_DATA_DIR=~/OSLdata \
    pytest tests/release -v -s

# Just one task:
RUN_OSL_RELEASE_TESTS=1 pytest tests/release/test_localization_release.py -v -s
RUN_OSL_RELEASE_TESTS=1 OSL_RELEASE_DATA_DIR=~/OSLdata \
    pytest tests/release/test_localization_release.py -v -s

# Just one backbone/config:
RUN_OSL_RELEASE_TESTS=1 pytest tests/release -v -s -k "rn18-gru"

# Skip the heaviest tests:
RUN_OSL_RELEASE_TESTS=1 pytest tests/release -v -s -m "not slow"

# True full-scale run on the fallback datasets (every file, e.g. the full
# ~111GB feature dataset) -- irrelevant once the OSL-ready primaries land,
# since those download a handful of shard files regardless of this setting:
RUN_OSL_RELEASE_TESTS=1 OSL_RELEASE_MAX_CLIPS=all OSL_RELEASE_MAX_GAMES=all \\
    pytest tests/release -v -s
```

Or use the convenience wrapper (same env-var gate, same defaults):

```bash
RUN_OSL_RELEASE_TESTS=1 scripts/run_release_tests.sh
```

## Tuning a run

### Preferring OSL-ready (sharded) datasets

The org publishes a curated set of datasets as parquet + webdataset shards —
a handful of large files per split — under the
[`OpenSportsLab/osl-ready-datasets`](https://huggingface.co/collections/OpenSportsLab/osl-ready-datasets)
collection, specifically so consumers don't have to pull thousands of
individual clips one at a time. Every dataset fixture in this suite calls
`_release_common.prefer_osl_ready_dataset(primary, fallback)`: it checks
whether the OSL-ready `primary` repo is populated yet and, if so, downloads
it via `_release_common.download_shard_split()` — a thin wrapper around
`opensportslib.tools.hf_transfer.download_dataset_split_from_hf(...,
download_format="parquet")`, which does
`snapshot_download(allow_patterns=[f"{split}/*"])` and converts the result
to a local OSL v2 JSON. Only if `primary` isn't populated yet does it fall
back to `fallback` — a known-good, currently-populated but non-sharded
dataset, downloaded with the file-count capping described below.

This means: no dataset-count-capping env var below has any effect once the
corresponding OSL-ready dataset is published — a sharded download is a
handful of files regardless of how many samples are in the split. The caps
only matter for the fallback path, which is what every fixture currently
uses (see the table above).

### Why fallback file counts are capped by default

The fallback datasets are loose per-clip/per-game files on the Hub (3450
files for the tennis dataset, 2210 for the SoccerNet feature dataset).
Downloading thousands of individual small files is slow and can trip
Hugging Face's API rate limit (1000 requests/5min on a free account — this
suite has hit it). So the fallback path in each fixture lists files first
(one cheap API call), then downloads only an annotation file plus a capped
number of referenced media files — never a blind "download the whole repo".

### Variables

Every test honors these environment variables so a run can be scaled from a
quick sanity pass up to a full release-scale run:

| Variable | Default | Meaning |
| --- | --- | --- |
| `OSL_RELEASE_CACHE_DIR` | `<repo>/.release_test_cache` | Where materialized configs and run outputs (checkpoints/logs/predictions) go. |
| `OSL_RELEASE_DATA_DIR` | `<cache dir>/data` | Where datasets are downloaded to / read from. Point this at a folder that already hosts the datasets (a shared drive, a previous run) and nothing gets re-downloaded or overwritten — huggingface_hub only fetches files that are missing or changed. Independent of `OSL_RELEASE_CACHE_DIR` so a large, reusable dataset cache can live separately from ephemeral run outputs. |
| `OSL_RELEASE_EPOCHS` | varies per test (1–3) | `TRAIN.epochs` override. `0` keeps each config's own default. |
| `OSL_RELEASE_MAX_CLIPS` | `40` | **Fallback only.** Cap on localization E2E clips downloaded per split (tennis dataset, 3450 files total). `all` forces the full dataset. No effect once `OSL-SNBAS` is published. |
| `OSL_RELEASE_MAX_GAMES` | `10` | **Fallback only.** Cap on games downloaded per split for the feature-based (CALF/NetVLAD++) localization tests (2210 files total). `all` downloads the full ~111GB dataset. No effect once `OSL-SoccerNet` is published. |
| `OSL_RELEASE_CLS_NUM_FRAMES` | `16` | `DATA.inputs.video.sampling.num_frames` override for the classification backbones. |
| `OSL_RELEASE_CLS_REPO` | `OpenSportsLab/OSL-XFoul` | Override the preferred (OSL-ready) classification dataset. |
| `OSL_RELEASE_CLS_FALLBACK_REPO` | `OpenSportsLab/OSL-cls-UEFA-fouls` | Override the fallback classification dataset. |
| `OSL_RELEASE_LOC_E2E_REPO` | `OpenSportsLab/OSL-SNBAS` | Override the preferred (OSL-ready) E2E localization dataset. |
| `OSL_RELEASE_LOC_E2E_FALLBACK_REPO` | `OpenSportsLab/soccernetpro-localization-tennis` | Override the fallback E2E localization dataset. |
| `OSL_RELEASE_LOC_FEATURES_REPO` | `OpenSportsLab/OSL-SoccerNet` | Override the preferred (OSL-ready) feature-based localization dataset. |
| `OSL_RELEASE_LOC_FEATURES_FALLBACK_REPO` | `OpenSportsLab/SoccerNet-ActionSpotting-Features` | Override the fallback feature-based localization dataset. |
| `OSL_RELEASE_VQA_REPO` | `OpenSportsLab/OSL-XFoul` | Override the VQA dataset (no fallback exists). |
| `HF_TOKEN` / `HUGGINGFACE_TOKEN` | unset | Auth for gated dataset repos (currently the classification fallback). |

## How configs are built

`opensportslib/configs/<task>/<name>.yaml` files are not self-contained — the
library layers them on top of `opensportslib/configs/default.yaml` and
`opensportslib/configs/<task>/default.yaml` at load time, but only when the
config path physically lives inside `opensportslib/configs/<task>/` (see
`opensportslib/core/config/loader.py::_compose_yaml_layers`). `_release_common.
materialize_config()` reuses the library's own `load_config()` to get that
real composition, deep-merges this suite's dataset/run overrides on top, and
writes the fully-resolved result under `OSL_RELEASE_CACHE_DIR/configs/` so it
loads as a plain standalone file. This means every release test trains with
the same defaults a real user gets from the canonical config, not a
hand-rolled approximation of them.

## Updating this suite after a release

If a release adds a new backbone, head, or task:

1. Add it to the relevant matrix (`MVNETWORK_BACKBONES`, `E2E_BACKBONES`/
   `E2E_HEADS`/`E2E_MATRIX`, etc.).
2. If it needs a new canonical config, make sure one exists under
   `opensportslib/configs/<task>/` first — these tests deliberately don't
   hand-roll model configs, they override the real ones.
3. If a dataset referenced here has changed shape (most likely: `OSL-XFoul`
   going from an empty placeholder to a real upload, or `OSL-cls-UEFA-fouls`'s
   label schema), fix the conversion helper called out in that test file's
   module docstring — each one names the exact function to look at first.
