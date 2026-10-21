"""Package a validated FineDiving build and publish it in one Hub commit.

Use --check-only to build and inspect the release without contacting the Hub.
The existing repository and access gate must already exist. History is retained.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def package(root, release):
    from opensportslib.tools.osl_json_to_parquet import convert_json_to_parquet
    root, release = Path(root), Path(release)
    report = json.loads((root/'reports/build.json').read_text())
    if not report.get('complete') or report.get('clips') != 3000 or report.get('frames') != 312256:
        raise ValueError('A fully validated 3,000-clip FineDiving build is required')
    release.mkdir(parents=True, exist_ok=True)
    for split, expected in [('train',1801), ('valid',450), ('test',749)]:
        document = json.loads((root/f'{split}.json').read_text())
        if len(document['data']) != expected:
            raise ValueError(f'Unexpected {split} count')
        convert_json_to_parquet(root/f'{split}.json', root, release/split,
                                shard_size='1GB', missing_policy='raise', overwrite=True)
        (release/'annotations').mkdir(exist_ok=True)
        shutil.copyfile(root/'annotations'/f'{split}.json', release/'annotations'/f'{split}.json')
    (release/'reports').mkdir(exist_ok=True)
    shutil.copyfile(root/'reports/build.json', release/'reports/build.json')
    card = '''---
pretty_name: FineDiving Precise Event Spotting
license: other
license_name: finediving-research-terms
license_link: https://github.com/xujinglin/FineDiving/blob/main/agreement/Release_Agreement.pdf
task_categories:
- video-classification
tags:
- sports
- diving
- action-spotting
- precise-event-spotting
- opensportslib
configs:
- config_name: default
  data_files:
  - split: train
    path: train/metadata.parquet
  - split: validation
    path: valid/metadata.parquet
  - split: test
    path: test/metadata.parquet
---

# FineDiving precise event spotting

Complete 3,000-clip, four-event-class spotting release prepared for OpenSportsLib
and T-DEED reproduction. The original FineDiving AQA taxonomy has 52 action
classes; this spotting protocol uses Entry, Som(s).Pike, Som(s).Tuck and Twist(s).

| Split | Clips |
|---|---:|
| Train | 1,801 |
| Validation | 450 |
| Test | 749 |

## Media and timing

Videos are FFV1 RGB-lossless Matroska files, packaged as OSL WebDataset TAR
shards with Parquet metadata. Every output frame was fully decoded and its RGB
pixels verified against the decoded source JPEG sequence. There are 312,256
frames in total. Original JPEGs remain available in `sources/frames.zip`.

**25 fps is an explicit conversion clock, not a measured camera frame rate.**
Upstream labels use `fps: -1`. Original zero-based event `frame` indices are
retained; `position_ms = frame * 40`. Each record retains the upstream record,
source JPEG paths, lossless-video checksum, and decoded-RGB checksum.
Use frame-based tolerances for the precise event spotting benchmark.

This replaces an older metadata-only conversion whose video inputs were absent,
FPS was zero, and event timestamps were all zero. That conversion remains in
repository history; it must not be used for reproduction. The `raw` branch
retains the earlier native source annotations.

## Loading

```python
from opensportslib.tools import download_dataset_split_from_hf
download_dataset_split_from_hf(
    repo_id="OpenSportsLab/OSL-loc-FineDiving", revision="main",
    split="test", output_dir="data/finediving", download_format="parquet",
)
```

Authentication and accepted repository access conditions are required. The
downloaded split contains a reconstructed OSL JSON manifest and unpacked videos.
Dataset Viewer rows contain OSL `sample_payload` JSON strings; they are not
direct Hugging Face `Video` columns. Original frame-indexed annotations are
under `annotations/`. See `reports/build.json` for validation and source hashes.

## Sources and terms

- Original dataset: https://github.com/xujinglin/FineDiving
- Spotting labels: https://github.com/arturxe2/T-DEED/tree/COMMIT/data/finediving
- Source frame mirror: https://huggingface.co/datasets/ZWJ666/FineDiving/tree/56adbfbc482ce5eda2aaa1033d7a3212613b6cfc
- The mirror traces the frame archive to OpenDataLab/FineDiving. The archive's
  SHA-256 is recorded in the build report.

FineDiving's research and attribution terms apply; the software repository's MIT
license does not replace the dataset release agreement. OpenSportsLab confirmed
permission to mirror the dataset and annotations for this publication. The
existing access gate is preserved. Cite Xu et al., FineDiving (CVPR 2022), Hong
et al., Spotting Temporally Precise, Fine-Grained Events in Video (ECCV 2022),
and Xarles et al., T-DEED (CVPR Workshops 2024).
'''.replace('COMMIT', report['source_commit'])
    (release/'README.md').write_text(card)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--release-root',type=Path,required=True)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--repo-id',default='OpenSportsLab/OSL-loc-FineDiving')
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--receipt',type=Path,required=True)
    args=parser.parse_args()
    report=package(args.data_root,args.release_root)
    from tools.convert.prepare_finediving_spotting import checksum
    if checksum(args.archive) != report['archive_sha256']:
        raise ValueError('Original frame archive does not match build receipt')
    (args.release_root/'sources').mkdir(exist_ok=True)
    dest=args.release_root/'sources/frames.zip'
    if not dest.exists():
        dest.symlink_to(args.archive.resolve())
    if args.check_only:
        print(json.dumps(report,indent=2))
        return
    from huggingface_hub import HfApi,CommitOperationAdd,CommitOperationDelete
    api=HfApi()
    info=api.dataset_info(args.repo_id)
    operations=[CommitOperationAdd(path_in_repo=p.relative_to(args.release_root).as_posix(),
                                   path_or_fileobj=str(p))
                for p in sorted(args.release_root.rglob('*')) if p.is_file()]
    obsolete={f'annotations-localization-{s}.json' for s in ['train','valid','test']}
    operations.extend(CommitOperationDelete(path_in_repo=p.rfilename)
                      for p in info.siblings if p.rfilename in obsolete)
    commit=api.create_commit(repo_id=args.repo_id,repo_type='dataset',revision='main',
                             parent_commit=info.sha,operations=operations,
                             commit_message='Publish complete frame-verified FineDiving spotting release')
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps({'repo':args.repo_id,'commit':commit.oid,
                                        'url':commit.commit_url,**report},indent=2)+'\n')
    print(commit.commit_url,flush=True)


if __name__ == '__main__':
    main()
