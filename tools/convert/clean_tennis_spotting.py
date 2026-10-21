"""Repair tennis media boundaries while preserving the official spotting labels.

Requires an independently restored video for any truncated sample. Resolution
is measured, never upscaled or inferred from the source annotation metadata.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lossless_video import write_lossless_frames
from prepare_finediving_spotting import checksum, write_json


def prepare(source_root, reference, output, repairs, workers=4):
    import av

    source_root, reference, output, repairs = map(Path, (source_root, reference, output, repairs))
    jobs, headers, native = [], {}, {}
    for split, ref_split in [('train','train'), ('validation','val'), ('test','test')]:
        document = json.loads((source_root/split/f'{split}.json').read_text())
        refs = json.loads((reference/f'{ref_split}.json').read_text())
        expected = {r['video']:r for r in refs}
        records = document.pop('data')
        if Counter(r['metadata']['video'] for r in records) != Counter(expected.keys()):
            raise ValueError(f'{split}: sample membership differs from benchmark')
        headers[split], native[split] = document, refs
        for row in records:
            ref = expected[row['metadata']['video']]
            fps = ref['fps']
            actual = Counter((e['label'],round(float(e['position_ms'])*fps/1000)) for e in row['events'])
            wanted = Counter((e['label'],e['frame']) for e in ref['events'])
            if actual != wanted:
                raise ValueError(f'{row["id"]}: annotation mismatch')
            jobs.append((split,row,ref))

    def one(job):
        split, original, ref = job
        row = copy.deepcopy(original)
        identity = ref['video']
        src = source_root/split/row['inputs'][0]['path']
        wanted = ref['num_frames']
        repair = repairs/f'{identity}.mkv'
        with av.open(str(src)) as c:
            stream = c.streams.video[0]
            declared, fps = stream.frames, stream.average_rate
            width, height = stream.width, stream.height
            if declared <= 0:
                declared = sum(1 for _ in c.decode(video=0))
        action = 'retained'
        if declared < wanted:
            if not repair.is_file():
                raise ValueError(f'{identity}: truncated media requires an original-source repair')
            src, action = repair, 'restored_original_source'
        elif declared > wanted:
            action = 'removed_trailing_frame'
        relative = f'media/{split}/{identity}' + ('.mkv' if action != 'retained' else src.suffix)
        dest = output/relative
        dest.parent.mkdir(parents=True,exist_ok=True)
        if action == 'removed_trailing_frame':
            receipt_path = output/'receipts'/f'{identity}.json'
            signature = {'source_sha256':checksum(src),'expected_frames':wanted}
            receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
            if not receipt or receipt.get('source') != signature or not dest.exists() or checksum(dest) != receipt.get('sha256'):
                def frames():
                    with av.open(str(src)) as c:
                        count = 0
                        for frame in c.decode(video=0):
                            if count == wanted:
                                break
                            yield frame.to_ndarray(format='rgb24')
                            count += 1
                        if count != wanted:
                            raise ValueError(f'{identity}: decoder returned too few frames')
                receipt = {**write_lossless_frames(frames(),dest,fps),'source':signature}
                write_json(receipt_path,receipt)
            digest = receipt['sha256']
        else:
            if dest.is_symlink():
                dest.unlink()
            if not dest.exists():
                try:
                    os.link(src.resolve(), dest)
                except OSError:
                    shutil.copy2(src, dest)
            digest = checksum(dest)
            with av.open(str(dest)) as c:
                stream = c.streams.video[0]
                width,height = stream.width,stream.height
                if action == 'restored_original_source' and sum(1 for _ in c.decode(video=0)) != wanted:
                    raise ValueError('Original-source repair has the wrong frame count')
        row['inputs'][0].update(path=relative,fps=float(fps))
        row['metadata'].update(width=width,height=height,num_frames=wanted,
                               source_width=original['metadata'].get('width'),
                               source_height=original['metadata'].get('height'),
                               media_cleanup=action,sha256=digest)
        # Preserve labels and timestamps exactly; add authoritative frame indices.
        for event in row['events']:
            event['frame'] = round(float(event['position_ms'])*ref['fps']/1000)
        return split,row

    splits = {s:[] for s in headers}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index,(split,row) in enumerate(pool.map(one,jobs),1):
            splits[split].append(row)
            if index%250 == 0 or index == len(jobs):
                print(f'Checked {index}/{len(jobs)} tennis clips',flush=True)
    counts = Counter()
    resolutions = sorted({(r['metadata']['width'],r['metadata']['height'])
                          for rows in splits.values() for r in rows})
    paper_resolution = all(w >= 398 and h >= 224 for w,h in resolutions)
    for split,rows in splits.items():
        counts.update(r['metadata']['media_cleanup'] for r in rows)
        document = headers[split]
        document.setdefault('metadata',{}).update(media_resolutions=resolutions,
            paper_resolution_satisfied=paper_resolution,cleanup='Frame boundaries repaired; source labels unchanged')
        write_json(output/f'{split}.json',{**document,'data':rows})
        write_json(output/'annotations'/f'{split}.json',native[split])
    report={'samples':len(jobs),'actions':dict(counts),'resolutions':resolutions,
            'paper_resolution_satisfied':paper_resolution,
            'note':'A separate acquisition at the paper resolution is still required for strict reproduction.'}
    write_json(output/'reports/build.json',report)
    return report


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source-root','reference','output','repairs']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    print(json.dumps(prepare(a.source_root,a.reference,a.output,a.repairs,a.workers),indent=2))
