# Tennis stroke classification clips

`tools/convert/tennis_to_classification.py` converts the local tennis spotting
dataset into short MP4s and ordinary OSL classification manifests. The selected
labels, in classifier order, are:

1. `far_court_serve`
2. `far_court_swing`
3. `near_court_serve`
4. `near_court_swing`

Bounce annotations are excluded. One selected event produces one labeled clip.
Each clip requests 0.5 seconds before and after the event. Its bounds are trimmed
at the midpoints to neighboring selected strokes and at video boundaries. The
requested windows therefore have variable lengths of at most one second;
encoded bounds differ by at most one source frame because cuts use frame
boundaries. Clips retain native source frames, resolution and presentation-time
spacing. There is no image resizing, FPS conversion, artificial frame padding or
audio. MP4s use lossy H.264
at CRF 18; pixel-exact reproduction is not claimed.

## Convert the downloaded data

Run from the repository root in the OSL environment. PyAV supplies decoding and
H.264 encoding; a standalone FFmpeg executable is not required.

```bash
conda activate osl
python tools/convert/tennis_to_classification.py \
  --source-root data/tennis-localization/swing-data \
  --output-root data/tennis-classification \
  --workers 2 --invalid-events skip
```

The converter reads `train/train.json`, `validation/validation.json` and
`test/test.json`. Video paths resolve relative to their respective source
manifest. Source files remain unchanged. Source event timestamps must be
video-relative milliseconds.

The original train and validation splits share 19 source matches. The default
`--split-policy match --validation-fraction 0.2 --seed 42` pools those two splits
and holds out four whole matches for validation. The existing test split stays
unchanged. Every source match is confined to one output split. For the supplied
local data, the resulting counts are:

| Output | Matches | Stroke clips |
| --- | ---: | ---: |
| `train.json` | 15 | 6,719 |
| `valid.json` | 4 | 934 |
| `test.json` | 9 | 9,858 |

The total is 17,511 clips. One truncated test source,
`Tennis_wimbledon_2019_mens_semifinal_federer_nadal_114169_114468`, decodes to
4.4 seconds but has selected events at 4.52, 5.36 and 7.20 seconds. Those three
events are excluded with their labels, source checksum, actual duration and reason
in `reports/excluded_events.json`. The explicit `--invalid-events skip` policy
permits only out-of-video exclusions; other conversion errors still stop the
build. Without that option, out-of-video events also stop the build.

Holding out a fraction of matches does not imply the same fraction of clips;
matches have different numbers of annotated events. The held-out matches and
per-class counts are recorded in `reports/build.json`. These new train/validation
splits differ from the published spotting benchmark. To preserve the source
splits for a separate comparison, pass `--split-policy preserve` and a different
output root; that retains their shared source matches.

Output layout:

```text
data/tennis-classification/
├── train.json
├── valid.json
├── test.json
├── clips/{train,valid,test}/*.mp4
├── receipts/*.json
└── reports/{build,excluded_events}.json
```

Each manifest has `task: action_classification`, the same four-class `action`
head, one video input and `labels.action.label` per record. Metadata retains the
source match, original split/record/event, source-video hash, source frame and PTS
bounds, event offset within the new clip, and measured output dimensions,
duration and frame count. Every encoded clip is decoded again to verify frame
count, PTS spacing, dimensions and that the target event lies inside the clip.

Clips and manifests publish atomically. Rerunning the same command verifies
completed clips against their checksum receipts and reuses them. Corrupt or
unfinished clips are regenerated. Changes to annotations, conversion settings
or converter implementation require a separate output root. The report becomes
`complete` only after all expected clips and all three manifests are published.
Training refuses incomplete builds.

## Publish the classification dataset

Use the OSL upload wrapper to publish Parquet metadata and WebDataset TAR shards,
then a detailed dataset README and provenance reports:

```bash
python tools/upload/upload_tennis_classification.py --check-only
hf auth login
python tools/upload/upload_tennis_classification.py
```

The default destination is `OpenSportsLab/OSL-cls-tennis-public`, revision `main`.
The check-only command validates clip checksums and metadata and writes
`data/tennis-classification/README.md` for review, with no Hub calls. The upload
command creates the repository if needed, uploads all three splits through
`opensportslib.tools.upload_dataset_as_parquet_to_hf`, then commits the README,
sanitized conversion report, exclusions and checksums. See the
[upload guide](../../tools/upload/README.md#tennis-stroke-classification-dataset)
for retry behavior and options.

## Train the skeleton/graph classifier

The training launcher uses the existing task wrapper and tennis classification
YAML. Its architecture is frozen detection/pose/tracking, followed by the existing
GIN player graph encoder, temporal pooling neck and classification head.
Skeletons are extracted online per clip; only action-model parameters train.

```bash
opensportslib setup --pyg
opensportslib setup --pose

# Validate manifests, media paths, match separation and configuration only.
python tools/train/train_tennis_skeleton.py --check-only

# Start training explicitly; conversion never starts it automatically.
python tools/train/train_tennis_skeleton.py --epochs 20 --batch-size 2
```

The launcher defaults to OpenCV-route/PyAV raw decoding, 20 maximum epochs and
batch size two. It keeps the classification preset's validation-loss selection
and early-stopping patience. Weighted resampling is disabled, so clips are sampled
without replacement. The existing training loader drops an incomplete last batch.
Metrics should include balanced accuracy
and macro-F1 because serves are less frequent than swings.

To explicitly cache exact-clip skeletons for subsequent epochs/experiments:

```bash
python tools/train/train_tennis_skeleton.py --epochs 20 --batch-size 2 \
  --cache-dir data/tennis-classification/pose-cache
```

The cache is separate from the MP4 conversion. Native clips can have fewer than
the configured 64 sampled frames; repeated sampling reuses skeleton observations
within the clip. Pose/tracking state resets independently for each clip.

For DALI, install the matching optional profile and add `--decoder dali` to the
training command. DALI execution remains subject to GPU validation and its CFR
input requirements. Training writes checkpoints under `checkpoints/tennis/` and
does not evaluate the held-out test set automatically. Use the existing
`ClassificationModel.infer` and `evaluate` APIs with the returned best checkpoint
when test evaluation is appropriate.

The required regression entry point is `bash scripts/run_tests.sh`. Offline
conversion tests use real encoded video, including fractional FPS/nonzero PTS,
edge events, class filtering, match-level splits, atomic resume and corrupt-clip
recovery. The launcher supports `--check-only` without constructing pose models.
