"""Publish the validated tennis cleanup to the existing swing-data branch."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import tarfile

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--release-root',type=Path,required=True)
    p.add_argument('--source-commit',required=True)
    p.add_argument('--expected-parent',help='Current branch commit when republishing a correction')
    p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--check-only',action='store_true')
    a=p.parse_args()
    from opensportslib.tools.osl_json_to_parquet import convert_json_to_parquet
    from huggingface_hub import HfApi,CommitOperationAdd,CommitOperationDelete
    report=json.loads((a.data_root/'reports/build.json').read_text())
    if report['samples']!=3445 or report['actions'].get('restored_original_source')!=1:
        raise ValueError('Expected the fully validated 3,445-clip cleanup')
    for split in ['train','validation','test']:
        convert_json_to_parquet(a.data_root/f'{split}.json',a.data_root,a.release_root/split,
                                shard_size='1GB',missing_policy='raise',overwrite=True)
        for archive in (a.release_root/split).rglob('*.tar'):
            with tarfile.open(archive) as tar:
                if any(not member.isfile() for member in tar):
                    raise ValueError(f'{archive}: all archive entries must contain regular file bytes')
    shutil.copytree(a.data_root/'annotations',a.release_root/'annotations',dirs_exist_ok=True)
    shutil.copytree(a.data_root/'reports',a.release_root/'reports',dirs_exist_ok=True)
    card='''---
pretty_name: Tennis Six-Class Event Spotting
license: other
task_categories:
- video-classification
tags:
- tennis
- action-spotting
- opensportslib
configs:
- config_name: default
  data_files:
  - split: train
    path: train/metadata.parquet
  - split: validation
    path: validation/metadata.parquet
  - split: test
    path: test/metadata.parquet
---

# Tennis six-class event spotting — swing-data

This branch contains the six original spotting classes: far/near-court bounce,
serve and swing. It contains 1,368 training, 455 validation and 1,622 test clips,
with 33,791 events. Sample membership, event labels and event frames match
T-DEED's released tennis annotations. It is distinct from a binary swing task.

## Cleanup and measured media properties

- 2,587 clips retained without re-encoding.
- 857 clips had one excess trailing frame; only that final frame was removed.
  The retained RGB frames were encoded losslessly with FFV1 and fully verified.
- The truncated Wimbledon 2019 Federer/Nadal test clip at source frames
  [114169,114468) was restored from the original YouTube video `wZnCcqm_g-E`.
  Its full 299-frame sequence replaces a broken 110-frame clip.
- Media dimensions are measured as **256 by 144**, replacing misleading
  source-video dimensions in the clip metadata. Original source dimensions
  remain separately recorded. Native event frame indices are retained.

**This 144p media release does not meet the paper's spatial-resolution setup.**
A higher-resolution acquisition is needed for strict T-DEED reproduction;
upscaling these videos would not restore the missing detail. The cleanup fixes
media boundaries and metadata while retaining the existing dataset resolution.

## Format and loading

OSL Parquet metadata and WebDataset TAR shards are provided for `train`,
`validation` and `test`. Videos are MP4 for retained assets and Matroska/FFV1
for repaired assets; the OSL loader supports both.

```python
from opensportslib.tools import download_dataset_split_from_hf
download_dataset_split_from_hf(
    repo_id="OpenSportsLab/OSL-loc-tennis-public", revision="swing-data",
    split="test", output_dir="data/tennis", download_format="parquet",
)
```

Repository authentication and accepted access conditions are required. The
original spotting annotations are under `annotations/`, and cleanup counts are
in `reports/build.json`. Historical state is retained in repository history and
the tag `swing-data-before-cleanup-2026-10-21`.

## Provenance

Source branch: `OpenSportsLab/OSL-loc-tennis-public@swing-data`, commit SOURCE.
Original benchmark: https://github.com/jhong93/spot/tree/main/data/tennis
T-DEED reference: https://github.com/arturxe2/T-DEED/tree/d953b4de8a2ad941c565e1683c5b07255cbe644e/data/tennis

Original dataset and source-media terms apply. This cleanup does not relicense
the broadcast videos. Cite Hong et al., Spotting Temporally Precise,
Fine-Grained Events in Video (ECCV 2022), and the original Vid2Player work.
'''.replace('SOURCE',a.source_commit)
    (a.release_root/'README.md').write_text(card)
    if a.check_only:return
    api=HfApi();repo='OpenSportsLab/OSL-loc-tennis-public';branch='swing-data'
    info=api.dataset_info(repo,revision=branch)
    if info.sha!=(a.expected_parent or a.source_commit):
        raise RuntimeError('The source branch changed; refusing to overwrite concurrent updates')
    api.create_tag(repo,tag='swing-data-before-cleanup-2026-10-21',revision=a.source_commit,
                   repo_type='dataset',exist_ok=True)
    files=[x for x in sorted(a.release_root.rglob('*')) if x.is_file()]
    paths={x.relative_to(a.release_root).as_posix() for x in files}
    operations=[CommitOperationAdd(path_in_repo=x.relative_to(a.release_root).as_posix(),path_or_fileobj=str(x)) for x in files]
    operations.extend(CommitOperationDelete(path_in_repo=x.rfilename) for x in info.siblings
                      if x.rfilename.startswith(('train/','validation/','test/')) and x.rfilename not in paths)
    commit=api.create_commit(repo_id=repo,repo_type='dataset',revision=branch,parent_commit=info.sha,
                             operations=operations,commit_message='Repair tennis clip boundaries and retain original six-class spotting annotations')
    a.receipt.parent.mkdir(parents=True,exist_ok=True)
    a.receipt.write_text(json.dumps({'repo':repo,'revision':branch,'commit':commit.oid,
                                     'url':commit.commit_url,**report},indent=2)+'\n')
    print(commit.commit_url,flush=True)


if __name__=='__main__':main()
