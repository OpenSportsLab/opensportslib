"""Publish complete FigureSkating sources and both frame-aligned protocols."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release-root',type=Path,required=True)
    p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--check-only',action='store_true')
    a=p.parse_args()
    report=json.loads((a.release_root/'reports/build.json').read_text())
    expected={'fs_comp/train':178,'fs_comp/valid':59,'fs_comp/test':134,
              'fs_perf/train':183,'fs_perf/valid':94,'fs_perf/test':94}
    if not report.get('complete') or report['protocols']!=expected or report['source_video_count']!=11 or report['unique_performances']!=371:
        raise ValueError('Both complete benchmark protocols and all 11 sources are required')
    card='''---
pretty_name: FigureSkating Precise Event Spotting
license: other
task_categories:
- video-classification
tags:
- figure-skating
- precise-event-spotting
- action-spotting
- opensportslib
configs:
- config_name: fs_comp
  data_files:
  - split: train
    path: metadata/fs_comp/train.parquet
  - split: validation
    path: metadata/fs_comp/valid.parquet
  - split: test
    path: metadata/fs_comp/test.parquet
- config_name: fs_perf
  data_files:
  - split: train
    path: metadata/fs_perf/train.parquet
  - split: validation
    path: metadata/fs_perf/valid.parquet
  - split: test
    path: metadata/fs_perf/test.parquet
---

# FigureSkating precise event spotting

The complete 371-performance benchmark over 11 original competition broadcasts,
with both official protocols and four event classes: jump takeoff/landing and
flying-spin takeoff/landing.

| Protocol | Train | Validation | Test |
|---|---:|---:|---:|
| FS-Comp | 178 | 59 | 134 |
| FS-Perf | 183 | 94 | 94 |

The protocols partition the same performances differently; media is shared and
stored only once. Do not combine their training and evaluation sets.

## Media and timing

Original 1920x1080, 25-fps source videos are in `media/`. Historical VP9 format
248 is used where available. The two women's World Championship broadcasts
from 2017 and 2018 use available H.264 format 137 at the same resolution and
frame rate, from the same original YouTube IDs. This encoding difference is
recorded; bit-identical RGB with historical VP9 input is not claimed.

Performance clips are represented by exact half-open source-frame intervals,
avoiding another video re-encoding. OSL event `position_ms` is absolute on the
source video; the loader rebases it through `metadata.intervals` to each logical
performance. `reference_frame` retains the native clip-relative event index,
and `source_frame` retains the absolute source frame. At 25 fps one frame is
exactly 40 ms. Native annotations remain in `annotations/fs_comp/` and
`annotations/fs_perf/`.

Each source has verified dimensions, FPS, interval bounds and a SHA-256 hash.
This does not constitute a full decode of every hour of source media. Exact
preprocessing and the selected decoder remain part of the experiment recipe.

## OpenSportsLib loading

Use the OpenCV backend for interval-aware localization. Current DALI localization
does not support these logical intervals.

```python
from opensportslib.tools import download_dataset_split_from_hf
download_dataset_split_from_hf(
    repo_id="OpenSportsLab/OSL-loc-FigureSkating", revision="main",
    split="fs_comp_test", output_dir="data/figureskating",
    download_format="json",
)
```

JSON manifests reference shared full broadcasts. Metadata-only reads use
`annotations_only=True`; selective input downloads are also supported. This
dataset uses shared raw media and OSL JSON rather than duplicating source videos
into separate per-protocol WebDataset archives. Parquet files index the samples
for the two Hugging Face configurations; `sample_payload` contains each OSL row.

## Sources and citation

- https://github.com/jhong93/spot/tree/main/data/fs_comp
- https://github.com/jhong93/spot/tree/main/data/fs_perf
- https://github.com/arturxe2/T-DEED/tree/COMMIT/data
- https://jhong93.github.io/projects/vpd.html

See `reports/build.json` for the immutable reference commit and source hashes.
Research access is manually gated. Source broadcast rights and annotation terms
remain with their respective owners; the source code license does not relicense
the videos. Cite Hong et al., Video Pose Distillation (ICCV 2021), Hong et al.,
Spotting Temporally Precise, Fine-Grained Events in Video (ECCV 2022), and
Xarles et al., T-DEED (CVPR Workshops 2024).
'''.replace('COMMIT',report['source_commit'])
    (a.release_root/'README.md').write_text(card)
    if a.check_only:return
    from huggingface_hub import HfApi,CommitOperationAdd
    from huggingface_hub.errors import RepositoryNotFoundError
    api=HfApi();repo='OpenSportsLab/OSL-loc-FigureSkating'
    try:
        before=api.dataset_info(repo)
    except RepositoryNotFoundError:
        api.create_repo(repo,repo_type='dataset',private=False)
        api.update_repo_settings(repo,repo_type='dataset',gated='manual')
        before=api.dataset_info(repo)
    if any(x.rfilename not in {'.gitattributes'} for x in before.siblings):
        raise RuntimeError('Target is already populated; refusing to overwrite it')
    operations=[CommitOperationAdd(path_in_repo=p.relative_to(a.release_root).as_posix(),path_or_fileobj=str(p))
                for p in sorted(a.release_root.rglob('*')) if p.is_file()]
    commit=api.create_commit(repo_id=repo,repo_type='dataset',revision='main',parent_commit=before.sha,
                             operations=operations,commit_message='Publish FigureSkating sources with FS-Comp and FS-Perf spotting protocols')
    a.receipt.parent.mkdir(parents=True,exist_ok=True)
    a.receipt.write_text(json.dumps({'repo':repo,'commit':commit.oid,'url':commit.commit_url,
                                     **report},indent=2)+'\n')
    print(commit.commit_url,flush=True)


if __name__=='__main__':main()
