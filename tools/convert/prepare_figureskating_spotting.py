"""Package FigureSkating protocols over shared, original source videos.

OSL intervals retain half-open source-frame bounds without re-encoding media.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from prepare_finediving_spotting import checksum,write_json


def prepare(source,reference,output,source_commit):
    import cv2
    import pyarrow as pa
    import pyarrow.parquet as pq

    source,reference,output=map(Path,(source,reference,output))
    files,media={},{}
    protocols={}
    performances=set()
    for protocol in ['fs_comp','fs_perf']:
        classes=(reference/protocol/'class.txt').read_text().splitlines()
        for split,native in [('train','train'),('valid','val'),('test','test')]:
            refs=json.loads((reference/protocol/f'{native}.json').read_text())
            records=[]
            for row in refs:
                performances.add(row['video'])
                base,_,start,end=row['video'].rsplit('_',3)
                start,end=int(start),int(end)
                if end-start!=row['num_frames']:
                    raise ValueError('Reference interval frame count mismatch')
                if base not in files:
                    matches=[p for p in source.glob(base+'.*') if p.suffix in ['.webm','.mp4','.mkv']]
                    if len(matches)!=1:
                        raise ValueError(f'Expected one complete original source: {base}')
                    p=matches[0]
                    capture=cv2.VideoCapture(str(p))
                    try:
                        fps=capture.get(cv2.CAP_PROP_FPS)
                        n=int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
                        dims=[int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))]
                    finally:
                        capture.release()
                    if abs(fps-25)>1e-6 or dims!=[1920,1080]:
                        raise ValueError(f'{base}: source format mismatch')
                    relative=f'media/{p.name}'
                    dest=output/relative;dest.parent.mkdir(parents=True,exist_ok=True)
                    if not dest.exists():dest.symlink_to(p.resolve())
                    files[base]=relative
                    media[base]={'path':relative,'sha256':checksum(p),'fps':fps,'frames':n,'dimensions':dims}
                if not 0<=start<end<=media[base]['frames']:
                    raise ValueError(f'{row["video"]}: interval outside source video')
                events=[]
                for event in row['events']:
                    if event['label'] not in classes or not 0<=event['frame']<end-start:
                        raise ValueError('Reference event outside performance')
                    events.append({**event,'head':'skating_action','reference_frame':event['frame'],
                                   'source_frame':start+event['frame'],
                                   'position_ms':(start+event['frame'])*40})
                records.append({'id':row['video'],'inputs':[{'type':'video','path':files[base],'fps':25}],
                    'metadata':{'video':row['video'],'source_start_frame':start,'source_end_frame':end,
                                'num_frames':row['num_frames'],'upstream_record':copy.deepcopy(row),
                                'intervals':[{'period':row['video'],'start_time_ms':start*40,'end_time_ms':end*40}]},
                    'events':events})
            doc={'version':'2.0','task':'action_spotting','dataset_name':'FigureSkating',
                 'metadata':{'protocol':protocol,'source_commit':source_commit,
                             'event_position_clock':'absolute source-video milliseconds; intervals rebase to clip time'},
                 'labels':{'skating_action':{'type':'single_label','labels':classes}},'data':records}
            write_json(output/f'{protocol}_{split}.json',doc)
            write_json(output/'annotations'/protocol/f'{native}.json',refs)
            folder=output/'metadata'/protocol;folder.mkdir(parents=True,exist_ok=True)
            pq.write_table(pa.Table.from_pylist([{'sample_id':r['id'],'sample_payload':json.dumps(r)} for r in records]),folder/f'{split}.parquet')
            protocols[f'{protocol}/{split}']=len(records)
    report={'complete':True,'source_commit':source_commit,'sources':media,'protocols':protocols,
            'source_video_count':len(files),'unique_performances':len(performances),
            'media_validation':'Source dimensions, FPS, interval bounds and file hashes; no full-video decode.'}
    write_json(output/'reports/build.json',report)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','reference','output','source-commit']:
        p.add_argument('--'+name,required=True)
    a=p.parse_args()
    print(json.dumps(prepare(a.source,a.reference,a.output,a.source_commit),indent=2))
