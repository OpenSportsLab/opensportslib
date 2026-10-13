# Full-rate tracking classification clips

`tools/convert/spotting_tracking_to_classification.py` creates one Parquet clip
per sample in the existing tracking classification dataset, retaining every native
tracking frame in the sample's window. It preserves sample IDs, splits, labels,
game IDs and action timestamps. This is a separate command from MP4 conversion;
it does not change an MP4 build or write into either supplied dataset.

The supplied YAML uses:

- Reference: `/home/giancos/OSLdata/sngar/tracking`.
- Source: `/home/giancos/OSLdata/PFF/SNGAR-Action-Spotting-Merged/main`.
- Output: `/home/giancos/OSLdata/sngar_tracking_classification_full_fps`.

## Run it

From the repository root, using the existing `osl` environment:

```bash
conda activate osl

python -u tools/convert/spotting_tracking_to_classification.py audit \
  --config tools/convert/spotting_tracking_to_classification.yaml

# Small pilot only: 18 samples covering all labels and splits.
python -u tools/convert/spotting_tracking_to_classification.py pilot \
  --config tools/convert/spotting_tracking_to_classification.yaml

# Run or resume all 87,939 samples after the pilot passes.
python -u tools/convert/spotting_tracking_to_classification.py build \
  --config tools/convert/spotting_tracking_to_classification.yaml
```

Alternatively, add `--then-build` to the pilot command to scale automatically.
No background job is started by the script. After changing configuration, inputs,
or implementation, rerun audit and pilot before building.

## Native-frame windows and gaps

Existing reference Parquets contain 16 rows sampled every nine `frameNum` values.
The converter reads those actual frame numbers and includes the entire interval:

```text
[first_reference_frame, last_reference_frame + 9)
```

The result is 144 consecutive native frame slots, approximately 4.805 seconds at
29.97 FPS. It does not infer time from row index, stretch gaps, interpolate player
positions, or downsample. Tracking and video extraction have different historical
center rounding rules; a tracking window can differ by one native frame from the
MP4 converter's window. Tracking reference frame numbers are authoritative here.

The source contains gaps, including periods before tracking starts and after it
ends. Choose their representation through `gap_policy`:

- **`pad` (default):** retain 144 rows. Missing rows have `tracking_valid=false`,
  the correct `frameNum`, null `videoTimeMs`, `period=-1`, and empty `[]`
  player/ball arrays. Other missing attributes remain null. These are missing
  slots, not invented tracking observations.
- **`observed`:** retain only real source rows. Clips can have fewer than 144 rows;
  frame numbers and timestamps retain their original gaps. `tracking_valid` is
  true for every retained row. Do not treat row index as elapsed time in this mode.

Observed source rows are copied without changing their values or column types.
All source columns are retained, plus the boolean `tracking_valid` column. Older
reference-only fields, such as the removed `video_url` field, are not fabricated.
The existing tracking parser handles empty player/ball arrays as missing objects.
Consumers that need a temporal validity mask can read `tracking_valid` explicitly.

The output manifest's FPS is estimated from the source `frameNum`/`videoTimeMs`
clock (rounded to three decimals, about 29.97), rather than copying the source
manifest's nominal 30 FPS. Real source timestamps remain unchanged and absolute.

## Validation and outputs

The audit verifies all action identities and metadata, reference frame schedules,
center alignment within 10 ms, and every observed reference timestamp against its
source within 1 ms. Source frame numbers/timestamps must be finite, unique and
increasing. It hashes local source and reference files; unchanged hashes can be
reused using size and nanosecond mtime. This script uses the supplied local copies
and does not make Hub verification or download claims.

The pilot validates Zstandard Parquet round trips and exact equality of all
observed rows with the source. It exercises leading/trailing gaps and reports
missing native-frame numbers, native FPS and source/reference hashes. Reports and
receipts record the reference Hub commit when present in the local manifest.

```text
<output>/
  tracking/
    train/clip_000000.parquet
    valid/clip_000000.parquet
    test/clip_000000.parquet
    pilot_{train,valid,test}.json
    annotations_{train,valid,test}.json   # written after full build completes
  reports/
    audit.json
    pilot.json
    build.json
    completed/<split>/<sample_id>.json
```

Output manifests use the existing `tracking_parquet` classification schema and
are checked with the existing annotation loader. Relative media paths resolve
from `<output>/tracking`. `metadata.conversion` contains native window bounds,
actual observed/missing counts, source timestamps, validation tolerances via the
configuration fingerprint, and the matching reference sample ID.

Conversion loads one source game at a time and writes clips immediately. Atomic
writes and checksummed completion receipts support resumption after interruption;
completed clips are reused only when their inputs/settings and hashes match.
Runtime estimates from the small pilot include loading several complete games
and can overestimate full-build runtime. The estimate includes a 10% free-space
margin. The full run is deliberately left for you to launch.

The supported test entry point is `bash scripts/run_tests.sh`. Offline tests cover
native frame expansion, beginning/internal gaps, both gap policies, unchanged
source values/types, tracking-parser compatibility, bad clocks, gating and recovery.

## Upload the completed dataset

Use [the tracking-full upload script](https://github.com/OpenSportsLab/opensportslib/blob/dev/tools/upload/README.md#soccernet-gar-full-fps-tracking-dataset)
to publish the final manifests and clips through OSL's Parquet/WebDataset uploader.
Run its `--check-only` mode first; it checks locally without contacting Hugging Face.
