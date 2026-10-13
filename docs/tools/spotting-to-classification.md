# Synchronized GAR classification MP4s from spotting

`tools/convert/spotting_to_classification.py` creates one native-FPS MP4 for every
sample in the existing classification release. IDs, splits, labels and action
timestamps provide the one-to-one mapping. The supplied YAML uses
`validation: temporal`: pixel differences are diagnostics, not conversion errors.
The original NumPy clips remain the reference and are not rewritten or duplicated.

Optional `validation: exact` retains the previous strict pixel-reproduction mode
and generates both regenerated NumPy arrays and MP4s only if exact equality passes.

Use the existing `osl` environment (NumPy, OpenCV, PyAV, PyYAML, PyArrow and
huggingface_hub). No standalone FFmpeg executable is needed. Existing Hugging Face
credentials are used for authenticated dataset downloads.

```bash
conda activate osl
python tools/convert/spotting_to_classification.py audit \
  --config tools/convert/spotting_to_classification.yaml
python tools/convert/spotting_to_classification.py pilot \
  --config tools/convert/spotting_to_classification.yaml --then-build
```

The YAML supplies the local spotting and reference roots, a separate output root,
Hub verification, pilot size (18), source-frame sampling interval (9), MP4 CRF
(18), validation mode (`temporal`), x264 preset (`veryfast`), build workers (2),
and expected split counts. Each encoder uses two threads; each decoder uses one. A direct Hub layout and the downloaded
`<split>/<split>/...` layout are both supported. The default output is
`/home/giancos/OSLdata/sngar_classification_rebuilt`.

## Verification and downloads

The source commits are fixed in the converter:

- Spotting: `a4a69ae4d40e706080183bf39fe5bf6f1d3a47ba`.
- Classification: `629043a5f3fe74e53095c7b8d3c0b489ff0d0269`.

The audit compares annotations and shard membership against the Hub, checks every
reference array's shape and dtype, and computes local SHA-256 hashes. Every source
MP4 and tracking Parquet is compared against its Hub SHA-256. Unchanged local
hashes can be reused on subsequent audits using file size and nanosecond mtime.

**Reference verification limitation:** the published classification shard manifest
contains no per-array checksums. Local array hashes establish reproducibility and
change detection, not independent equality with the remote tar member. The audit
reports this distinction explicitly. It does not download all tar shards just to
re-hash an already extracted copy.

Missing/stale manifests and missing or invalid media are downloaded into the
separate `verified_inputs`/`hub_cache` trees; the supplied copies are never changed.
Selective reference repair uses the existing OSL downloader, which may download
one complete shard and extract its other missing members too. Pixel-valid arrays
that differ from their remote tar members cannot be detected by metadata alone.
Setting `verify_hub: false` is for offline fixtures or an explicitly local audit;
it is recorded in provenance and is not the supplied release configuration.

## Pilot and validation modes

Selection is deterministic (seed 42), covers all labels, the first/last sample of
each split, and additional samples up to the configured count. Run `pilot` without
`--then-build` to produce just the 18 sample videos for inspection.

In **temporal mode**, clip indices follow the historical extraction rule:

```text
center = floor(position_ms / 1000 * source_fps)
sampled_frames = [center + (i - 8) * stride for i in range(16)]
MP4 frames = [sampled_frames[0], sampled_frames[-1] + stride)
```

This rule is fixed for the release; pixel error never chooses a different offset
for individual samples. All 87,939 action windows are checked for valid bounds
before pilot MP4 generation. Every generated MP4 must preserve contiguous native
frames, source dimensions, frame timestamps and duration. Its annotated action
must occur inside the clip and within one native frame of sample index 8, allowing
only timestamp quantization. Frame-to-frame pixel differences are recorded for
the pilot but do not reject synchronized video output. These checks establish
alignment to the published action timestamps and historical sampling schedule;
they do not claim recovery of undocumented original frame preprocessing.

In **exact mode**, the pilot tests 100 bounded historical/diagnostic profiles:
source or declared 30-fps indexing, offsets 0/±1/±2, RGB/BGR, and
linear/area/cubic/Lanczos resizing or a linear 398×224 intermediate resize. Every
pilot array must match the `(16, 224, 224, 3)` `uint8` reference exactly using one
common profile. This optional mode still exits with status 2 if equality fails.

`reports/pilot.json` records the validation mode, diagnostics, timing checks,
full-dataset preflight count and size/runtime estimates. Contact sheets show
reference, candidate, and amplified absolute error for sampled frames 0, 8 and 15.
Timing or event-mapping failures block both modes. The historical 224p preprocessing
mismatch no longer blocks temporal mode.

## Video and output contract

MP4 clips cover the full sampling window:
`[first_sample, last_sample + stride)`. At stride 9, all 144 native frames are
included, lasting about 4.805 seconds at 30000/1001 fps. Clips preserve source
resolution, PTS deltas and fractional rate, with H.264 CRF 18, the configured x264 preset, `yuv420p`, and no
audio. MP4 encoding is lossy; exact pixel equality is required only in exact mode.

The encoder seeks using the source stream's timestamp offset, checks contiguous
native indices, rebases PTS, and validates decoded output dimensions, frame count,
timestamps and duration. Irregular timestamps that cannot be mapped unambiguously
to native indices, missing frames and out-of-bounds windows fail instead of being
silently padded, stretched, or resampled.

Temporal outputs are `video/<split>/clip_*.mp4` and OSL
`video/annotations_<split>.json` manifests. Exact mode additionally writes the
`frames` tree. Pilot manifests use `pilot_<split>.json`. IDs, order, labels, splits
and original metadata are retained; `metadata.conversion` records the matching
reference sample ID, validation mode, source hashes/revisions, settings, sampled
indices, source timing, clip bounds and sampled clip timestamps. Temporal mode
also records the action's clip-relative timestamp and measured center timing
error. The existing annotation loader validates each emitted manifest. These
outputs do not inherit misleading `hf_*` download metadata from the reference.

`--then-build` scales automatically after a passing pilot. You can also run
`build --config ...` later. Estimates use measured pilot size and serial encoding
time, with a 10% disk-space margin; bounded parallel build workers can reduce the
actual runtime. The configured two workers limit competition with other jobs.
Manifest writing streams completion records. Both modes stop on timing or source
integrity failures; exact mode additionally stops on pixel mismatch.

Atomic writes and per-sample receipts make interrupted builds resumable.
Source/configuration/implementation/library fingerprints and output checksums
prevent incompatible or incomplete work being reused. Progress is saved every
ten completed clips in `reports/build.json`. Existing input files must remain
unchanged while a command runs.

Run the supported verification entry point with `bash scripts/run_tests.sh`.
The offline tests create small real videos and exercise matching, path handling,
fractional rates, nonzero PTS offsets, exactness failures, manifests and recovery.

For the equivalent full-rate tracking Parquet workflow, see
[tracking conversion](spotting-tracking-to-classification.md).

## Upload the completed videos

After the full conversion finishes, use
[the videos upload script](https://github.com/OpenSportsLab/opensportslib/blob/dev/tools/upload/README.md#soccernet-gar-native-fps-videos-dataset)
to package MP4s with OSL Parquet metadata and WebDataset TAR shards on the
`videos` branch. Start with `--check-only` for local validation without Hub access.

## Resume after the late-game timestamp failure

The older encoder mapped PTS to frame numbers using the declared codec rate.
Some source games declare `30000/1001` but measure approximately `29.97` FPS;
after more than two hours that difference exceeds its quarter-frame tolerance.
The encoder now maps indices using the stream's measured average rate and still
preserves the original PTS deltas, contiguous frames and action alignment. It
does not resample or force a constant frame rate.

The known previous converter's successful audit/pilot and checksum-valid outputs
are compatible with this fix, provided configuration, manifests and input files
are unchanged. Receipts are upgraded during resume without re-encoding valid
clips. Other converter identities do not receive this compatibility exception.

With no conversion process running:

```bash
python tools/convert/spotting_to_classification.py repair \
  --config tools/convert/spotting_to_classification.yaml
python tools/convert/spotting_to_classification.py build \
  --config tools/convert/spotting_to_classification.yaml
```

`repair` checks generated files against their successful conversion receipts,
removes incomplete temporary files, checksum-invalid outputs and outputs without
valid receipts, and writes `reports/repair.json`. It does not modify source data.
Matching hashes establish that completed clips are unchanged since the converter's
full decode validation. The resumed build regenerates missing/removed samples.
