# Spotting dataset preparation

The dataset cleanup tools preserve the published spotting splits and native
event frame indices. A matching annotation manifest does not prove that its
media retains the original temporal padding or spatial resolution.

## Canonical repositories

| Dataset | Repository | Revision/protocol |
|---|---|---|
| SoccerNet BAS | `OpenSportsLab/OSL-SNBAS` | `720p-2024` for the high-resolution 2024 benchmark |
| SoccerNet AS | `SoccerNet/SoccerNet_raw_HQ` | Official `videos-224p` or `videos-720p` media, according to the experiment; retain the existing OSL annotation manifests and pinned benchmark splits |
| Tennis | `OpenSportsLab/OSL-loc-tennis-public` | `swing-data`, six classes |
| FineDiving | `OpenSportsLab/OSL-loc-FineDiving` | `main` |
| FineGym | `OpenSportsLab/OSL-loc-gymnastics` | `main`; inspect the build report for completeness |
| FigureSkating | `OpenSportsLab/OSL-loc-FigureSkating` | `fs_comp` and `fs_perf` protocols |

FineDiving and gymnastics were renamed from
`soccernetpro-localization-diving` and
`soccernetpro-localization-gymnastics`. Their history and `raw` branches are
retained. Publication requires existing repository access and appropriate data
redistribution permissions; software licenses do not replace source data terms.

## Reference labels and auditing

Pin the upstream T-DEED repository to commit
`d953b4de8a2ad941c565e1683c5b07255cbe644e`. Do not create new random splits.

```bash
python tools/convert/audit_spotting_dataset.py \
  --manifest data/tennis/train/metadata.parquet \
  --reference reference/T-DEED/data/tennis/train.json \
  --output outputs/audit-tennis-train.json
```

The audit checks sample membership, duplicate IDs, original event labels and
frame indices, frame bounds and declared frame counts. It reports incomplete
coverage with a failing exit code. Media validation is a separate operation.

## FineDiving

The original trimmed-frame archive has SHA-256
`db1d034753f60c3022d4938fd63a5c5c621c8a341bf70cd45bcfc1f59bcb7582`.
Its 3,000 clips contain 312,256 JPEG frames. The source mirror and its provenance
are recorded on the published dataset card.

```bash
python tools/convert/prepare_finediving_spotting.py \
  --archive FineDiving_trimmed_frames_with_annotations.zip \
  --reference reference/T-DEED/data/finediving \
  --output data/finediving/prepared \
  --source-commit d953b4de8a2ad941c565e1683c5b07255cbe644e \
  --archive-sha256 db1d034753f60c3022d4938fd63a5c5c621c8a341bf70cd45bcfc1f59bcb7582
```

Preparation uses FFV1 RGB-lossless Matroska and fully decodes every output to
verify RGB pixel hashes. Original frame paths and upstream records are retained.
**25 fps is an explicit conversion clock, not measured camera FPS**: source
labels use `fps: -1`. Native event indices remain authoritative.

`tools/upload/upload_finediving_spotting.py` packages the validated output as OSL
Parquet/WebDataset, includes the original frame ZIP, and publishes in one commit.
Use `--check-only` to package without Hub writes. Local source videos and
receipts must remain unchanged while publishing.

## Tennis

The `swing-data` annotations match the 3,445-sample six-class benchmark. The
existing media is 256x144, which does not satisfy the paper's spatial setup.
The cleanup removes one excess trailing frame from 857 clips and replaces one
110-frame corrupted clip with its complete original 299-frame source segment.
The remaining 2,587 clips are retained. Labels and timestamps are unchanged;
authoritative event frame indices and measured dimensions are added.

`tools/convert/clean_tennis_spotting.py` requires an independently restored source
clip in `--repairs` for truncated media. It refuses to fabricate missing frames.
`tools/upload/upload_tennis_spotting_cleanup.py` publishes to `swing-data` with a
tag retaining the earlier revision. A high-resolution acquisition remains a
separate requirement for strict paper reproduction.

## FineGym

The legacy conversion covered only 422 of 5,374 benchmark clips. Its media also
lost temporal padding: an examined clip had 149 frames instead of 266. The full
source archive is available at
<https://opendatalab.com/OpenDataLab/FineGym>.

```bash
OPENCV_FFMPEG_THREADS=1 python -u tools/download/rebuild_finegym_from_opendatalab.py \
  --reference reference/T-DEED/data/finegym \
  --output data/finegym/prepared \
  --source-root data/finegym/source-staging \
  --source-commit d953b4de8a2ad941c565e1683c5b07255cbe644e \
  --workers 8 --publish
```

The six-part archive totals approximately 634 GB. The tool streams verified
byte ranges with bounded prefetch, processes source videos incrementally, and
removes only its downloaded staging videos after validation. It preserves the
reference padding and FPS stride, resizes to 224x224 with linear interpolation,
then verifies the RGB-lossless output. Original video hashes and source records
remain in receipts. This explicit resize/encoding choice does not claim
bit-identical pixels with an earlier JPEG preprocessing run. Some archive
sources are 720p rather than the 1080p recorded upstream; both dimensions are
retained. Sources must preserve the aspect ratio and exceed the 224-pixel input
size. Frame-rate mismatches remain errors rather than implicit resampling.

`reports/progress.json` records running, failed, interrupted or complete status.
Clip receipts permit reuse of verified outputs on restart. Re-reading an archive
prefix may still be necessary after interruption. All 5,374 clips must pass
before publication. A concurrent remote change blocks replacement of `main`.
Generating the WebDataset release requires additional local disk space beyond
the prepared clips. Complete archive-part checksums are not claimed verified
when streaming stops after obtaining all required sources.

## FigureSkating

Obtain the 11 original broadcasts listed in E2E-Spot's `data/fs_comp/videos.csv`.
Use 1080p/25-fps format 248 where available; the two women's World Championship
broadcasts from 2017 and 2018 currently offer H.264 format 137 at the same
resolution and FPS. Record the encoding difference and source hashes.

```bash
python tools/convert/prepare_figureskating_spotting.py \
  --source data/figureskating/source \
  --reference reference/T-DEED/data \
  --output data/figureskating/release \
  --source-commit d953b4de8a2ad941c565e1683c5b07255cbe644e
```

Original media is shared by both protocols. Root OSL JSON files such as
`fs_comp_test.json` describe half-open source intervals and absolute source-video
event timestamps. The loader rebases them to performance-relative time. Native
reference labels and per-protocol Parquet indexes are also included. Use the
OpenCV backend; the current DALI path does not support logical intervals.
`tools/upload/upload_figureskating_spotting.py` requires both complete protocols
and refuses to overwrite an already populated target.

These scripts do not change task APIs or prediction-saving behavior. Their
regression tests exercise actual encoding/decoding, corrupt-media recovery,
padding, frame-rate checks and interval rebasing through the library.
