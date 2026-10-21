"""Build lossless, frame-aligned OSL videos from FineDiving's trimmed JPEG ZIP.

The 25-fps output is an explicit conversion clock, not a measured source FPS.
Original event indices, upstream records and source JPEG paths are retained.
Every encoded clip is fully decoded and compared to its source RGB pixels.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import threading
import zipfile


def checksum(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.part')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def frame_groups(archive):
    groups = defaultdict(list)
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            path = PurePosixPath(member.filename)
            if path.suffix.lower() != '.jpg':
                continue
            if path.is_absolute() or '..' in path.parts or len(path.parts) != 4:
                raise ValueError(f'Unexpected JPEG path: {member.filename}')
            identity = '__'.join(path.parts[-3:-1])
            groups[identity].append(member.filename)
    for identity, names in groups.items():
        names.sort()
        indices = [int(PurePosixPath(name).stem) for name in names]
        if len(set(names)) != len(names) or indices != list(range(indices[0], indices[0]+len(indices))):
            raise ValueError(f'Duplicate or nonconsecutive source frames: {identity}')
    return dict(groups)


def encode_clip(source, names, output, fps):
    import numpy as np
    from PIL import Image
    # Support both script execution and imports from other conversion tools.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from lossless_video import write_lossless_frames

    def frames():
        for name in names:
            with Image.open(io.BytesIO(source.read(name))) as image:
                yield np.array(image.convert('RGB'))
    return write_lossless_frames(frames(), output, fps)


def prepare(archive, reference, output, *, workers=4, fps=25, source_commit,
            expected_archive_sha256=None):
    archive, reference, output = map(Path, (archive, reference, output))
    if fps <= 0 or workers <= 0:
        raise ValueError('Positive conversion FPS and worker count required')
    archive_hash = checksum(archive)
    if expected_archive_sha256 and archive_hash != expected_archive_sha256:
        raise ValueError('Source archive checksum mismatch')
    groups = frame_groups(archive)
    classes = (reference/'class.txt').read_text().splitlines()
    jobs, seen, references = [], set(), {}
    for split, upstream_split in [('train','train'), ('valid','val'), ('test','test')]:
        rows = json.loads((reference/f'{upstream_split}.json').read_text())
        references[split] = rows
        for row in rows:
            identity = row['video']
            if identity in seen:
                raise ValueError(f'Duplicate sample across splits: {identity}')
            seen.add(identity)
            if len(groups.get(identity, [])) != row['num_frames']:
                raise ValueError(f'Source frame count mismatch: {identity}')
            for event in row['events']:
                if event['label'] not in classes or not 0 <= event['frame'] < row['num_frames']:
                    raise ValueError(f'Invalid reference event: {identity}')
            jobs.append((split, row))
    if seen != set(groups):
        raise ValueError('Source clips do not exactly match the reference splits')
    thread = threading.local()

    def build(job):
        split, row = job
        identity = row['video']
        names = groups[identity]
        relative = f'media/{split}/{identity}.mkv'
        video = output/relative
        receipt_path = output/'receipts'/f'{identity}.json'
        signature = {'archive_sha256': archive_hash, 'source_commit': source_commit,
                     'conversion_fps': fps, 'source_frames': names,
                     'reference_sha256': hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()}
        receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
        if not receipt or receipt.get('source') != signature or not video.exists() or checksum(video) != receipt.get('sha256'):
            if not hasattr(thread, 'archive'):
                thread.archive = zipfile.ZipFile(archive)
            receipt = {**encode_clip(thread.archive, names, video, fps), 'source': signature}
            write_json(receipt_path, receipt)
        events = [{**event, 'head': 'diving_action',
                   'position_ms': round(event['frame'] * 1000 / fps)} for event in row['events']]
        record = {'id': f'FineDiving_{identity}',
                  'inputs': [{'type':'video', 'path':relative, 'fps':fps}],
                  'metadata': {'video':identity, 'num_frames':row['num_frames'],
                               'width':receipt['width'], 'height':receipt['height'],
                               'conversion_clock_fps':fps, 'source_fps':None,
                               'source_frame_paths':names, 'upstream_record':row,
                               'sha256':receipt['sha256'], 'rgb_sha256':receipt['rgb_sha256']},
                  'events': events}
        return split, record

    splits = defaultdict(list)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, (split, record) in enumerate(pool.map(build, jobs), 1):
            splits[split].append(record)
            if index % 100 == 0 or index == len(jobs):
                print(f'Validated {index}/{len(jobs)} clips', flush=True)
    for split, records in splits.items():
        write_json(output/f'{split}.json', {
            'version':'2.0', 'task':'action_spotting', 'dataset_name':'FineDiving',
            'metadata': {'source_commit':source_commit, 'source_archive_sha256':archive_hash,
                         'conversion_clock_fps':fps, 'source_fps_known':False},
            'labels':{'diving_action':{'type':'single_label','labels':classes}}, 'data':records})
        write_json(output/'annotations'/f'{split}.json', references[split])
    report = {'complete':True, 'clips':len(jobs), 'frames':sum(len(v) for v in groups.values()),
              'splits':{k:len(v) for k,v in splits.items()}, 'archive_sha256':archive_hash,
              'source_commit':source_commit, 'conversion_clock_fps':fps,
              'validation':'Every encoded frame decoded and RGB hash matched to source JPEG decode'}
    write_json(output/'reports/build.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-commit',required=True)
    parser.add_argument('--archive-sha256')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--fps',type=int,default=25)
    args=parser.parse_args()
    print(json.dumps(prepare(args.archive,args.reference,args.output,workers=args.workers,
                             fps=args.fps,source_commit=args.source_commit,
                             expected_archive_sha256=args.archive_sha256),indent=2))


if __name__ == '__main__':
    main()
