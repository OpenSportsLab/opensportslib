"""Rebuild FineGym spotting clips from the original, frame-compatible videos.

Each source video is processed independently, with resumable clip receipts.
Final manifests are written only after every benchmark clip has been verified.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
import json
import math
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from lossless_video import write_lossless_frames
from prepare_finediving_spotting import checksum,write_json


def reference_rows(reference):
    reference=Path(reference)
    return [(split,row) for split,native in [('train','train'),('valid','val'),('test','test')]
            for row in json.loads((reference/f'{native}.json').read_text())]


def prepare_source(video, reference, output, source_commit, *, workers=4):
    import cv2
    video,reference,output=map(Path,(video,reference,output))
    identity=video.stem
    jobs=[(split,row) for split,row in reference_rows(reference)
          if row['video'].split('_E_')[0]==identity]
    if not jobs:
        raise ValueError(f'Unexpected source video: {identity}')
    source_hash=checksum(video)
    probe=cv2.VideoCapture(str(video))
    try:
        fps=probe.get(cv2.CAP_PROP_FPS)
        total=int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
        dimensions=(int(probe.get(cv2.CAP_PROP_FRAME_WIDTH)),int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        probe.release()
    if fps<=0 or total<=0:
        raise ValueError('Invalid source video')

    def one(job):
        split,row=job
        reference_aspect=row['width']/row['height']
        if min(dimensions)<224 or not math.isclose(dimensions[0]/dimensions[1],reference_aspect,abs_tol=0.005,rel_tol=0):
            raise ValueError(f'{identity}: source spatial format cannot reproduce the input geometry')
        if math.isclose(fps,row['fps'],abs_tol=0.001,rel_tol=0):
            stride=1
        elif math.isclose(fps/2,row['fps'],abs_tol=0.001,rel_tol=0):
            stride=2
        else:
            raise ValueError(f'{identity}: source FPS {fps} differs from reference {row["fps"]}')
        q=row['_source_info']
        start=max(0,q['start_frame']-q['pad'][0])
        end=min(total,q['end_frame']+q['pad'][1])
        if math.ceil((end-start)/stride)!=row['num_frames']:
            raise ValueError(f'{row["video"]}: padded source boundaries have the wrong frame count')
        relative=f'media/{split}/{row["video"]}.mkv'
        dest=output/relative
        receipt_path=output/'receipts'/f'{row["video"]}.json'
        signature={'source_sha256':source_hash,'source_commit':source_commit,
                   'source_start_frame':start,'source_end_frame':end,'stride':stride,
                   'reference':row,'resize':[224,224]}
        receipt=json.loads(receipt_path.read_text()) if receipt_path.exists() else None
        if not receipt or receipt.get('source')!=signature or not dest.exists() or checksum(dest)!=receipt.get('sha256'):
            def frames():
                capture=cv2.VideoCapture(str(video))
                try:
                    capture.set(cv2.CAP_PROP_POS_FRAMES,start)
                    for index in range(end-start):
                        ok,bgr=capture.read()
                        if not ok:
                            raise ValueError(f'{row["video"]}: source decoding stopped early')
                        if index%stride==0:
                            resized=cv2.resize(bgr,(224,224),interpolation=cv2.INTER_LINEAR)
                            yield cv2.cvtColor(resized,cv2.COLOR_BGR2RGB)
                finally:
                    capture.release()
            receipt={**write_lossless_frames(frames(),dest,Fraction(row['fps']).limit_denominator(1_000_000)),
                     'source':signature}
            if receipt['num_frames']!=row['num_frames']:
                raise ValueError('Encoded clip has the wrong frame count')
            write_json(receipt_path,receipt)
        record={'id':f'FineGym_{row["video"]}',
                'inputs':[{'type':'video','path':relative,'fps':row['fps']}],
                'metadata':{'video':row['video'],'num_frames':row['num_frames'],
                            'width':224,'height':224,'source_video':identity,
                            'source_start_frame':start,'source_end_frame':end,'source_stride':stride,
                            'source_dimensions':dimensions,
                            'reference_source_dimensions':[row['width'],row['height']],
                            'source_sha256':source_hash,'upstream_record':row,
                            'sha256':receipt['sha256'],'rgb_sha256':receipt['rgb_sha256']},
                'events':[{**event,'head':'gymnastics_action',
                           'position_ms':round(event['frame']*1000/row['fps'])} for event in row['events']]}
        write_json(output/'records'/split/f'{row["video"]}.json',record)
        return row['video']
    with ThreadPoolExecutor(max_workers=workers) as pool:
        completed=list(pool.map(one,jobs))
    report={'source_video':identity,'source_sha256':source_hash,'source_fps':fps,
            'source_dimensions':dimensions,'clips':len(completed),'complete':True,
            'reference_dimensions':sorted({(row['width'],row['height']) for _,row in jobs}),
            'source_commit':source_commit,'clip_ids':completed}
    write_json(output/'source-receipts'/f'{identity}.json',report)
    return report


def finalize(reference,output,source_commit):
    reference,output=map(Path,(reference,output))
    rows=reference_rows(reference)
    missing=[row['video'] for split,row in rows if not (output/'records'/split/f'{row["video"]}.json').is_file()]
    if missing:
        write_json(output/'reports/missing.json',{'complete':False,'missing_samples':missing})
        raise ValueError(f'Missing {len(missing)} benchmark clips; refusing a complete release')
    classes=(reference/'class.txt').read_text().splitlines()
    for split,native in [('train','train'),('valid','val'),('test','test')]:
        records=[]
        for s,row in rows:
            if s!=split:continue
            record=json.loads((output/'records'/s/f'{row["video"]}.json').read_text())
            source_receipt=json.loads((output/'source-receipts'/f'{row["video"].split("_E_")[0]}.json').read_text())
            record['metadata'].setdefault('source_dimensions',source_receipt['source_dimensions'])
            record['metadata'].setdefault('reference_source_dimensions',[row['width'],row['height']])
            if checksum(output/record['inputs'][0]['path'])!=record['metadata']['sha256']:
                raise ValueError(f'Changed media: {row["video"]}')
            records.append(record)
        write_json(output/f'{split}.json',{'version':'2.0','task':'action_spotting','dataset_name':'FineGym',
                    'metadata':{'source_commit':source_commit,'resize':[224,224],
                                'validation':'Every encoded clip fully decoded and RGB verified'},
                    'labels':{'gymnastics_action':{'type':'single_label','labels':classes}},'data':records})
        write_json(output/'annotations'/f'{split}.json',json.loads((reference/f'{native}.json').read_text()))
    report={'complete':True,'clips':len(rows),'frames':sum(row['num_frames'] for _,row in rows),
            'source_commit':source_commit,'splits':{s:sum(k==s for k,_ in rows) for s in ['train','valid','test']}}
    write_json(output/'reports/build.json',report)
    return report
