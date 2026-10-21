"""Compare an OSL spotting release with pinned, frame-indexed benchmark labels.

This is read-only: mismatches are reported, never repaired or silently excluded.
Supports OSL JSON and OSL's sample_payload Parquet metadata. Media availability
is checked separately because a valid manifest does not prove media completeness.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path


def load_records(path):
    path = Path(path)
    if path.suffix == '.parquet':
        import pyarrow.parquet as pq
        rows = pq.read_table(path).to_pylist()
        if not rows or 'sample_payload' not in rows[0]:
            raise ValueError('Expected OSL sample_payload metadata')
        return [json.loads(row['sample_payload']) for row in rows]
    document = json.loads(path.read_text())
    if not isinstance(document, dict) or not isinstance(document.get('data'), list):
        raise ValueError('Expected an OSL document with a data list')
    return document['data']


def compare(records, reference):
    """Report coverage and exact label/frame agreement without altering data."""
    expected = {row['video']: row for row in reference}
    if len(expected) != len(reference):
        raise ValueError('Duplicate reference video IDs')
    seen = Counter()
    issues = []
    matched_events = 0
    labels = Counter()
    for row in records:
        inputs = row.get('inputs', [])
        metadata = row.get('metadata', {})
        video = metadata.get('video')
        if not video:
            video = Path(inputs[0]['path']).stem if inputs else row.get('id')
        seen[video] += 1
        target = expected.get(video)
        if target is None:
            issues.append({'video': video, 'issue': 'unexpected_sample'})
            continue
        fps = target.get('fps', 0)
        if not isinstance(fps, (int, float)) or fps <= 0:
            fps = metadata.get('fps_raw') or (inputs[0].get('fps') if inputs else None)
        events = []
        for event in row.get('events', []):
            labels[event.get('label')] += 1
            frame = event.get('reference_frame', event.get('frame'))
            position = event.get('position_ms')
            if position is not None:
                if not math.isfinite(float(position)):
                    issues.append({'video': video, 'issue': 'invalid_position'})
                    continue
                position = float(position)
                intervals = metadata.get('intervals')
                if intervals:
                    if len(intervals) != 1:
                        raise ValueError('Audit each logical interval separately')
                    position -= intervals[0]['start_time_ms']
                if position < 0:
                    issues.append({'video': video, 'issue': 'invalid_position'})
                    continue
                if frame is not None and fps and float(fps) > 0 and round(position * float(fps) / 1000) != frame:
                    issues.append({'video': video, 'issue': 'position_frame_mismatch'})
            if frame is None:
                if not fps or not math.isfinite(float(fps)) or float(fps) <= 0:
                    issues.append({'video': video, 'issue': 'unknown_fps_without_frame'})
                    continue
                if position is None or not math.isfinite(float(position)):
                    issues.append({'video': video, 'issue': 'invalid_position'})
                    continue
                frame = round(float(position) * float(fps) / 1000)
            if not isinstance(frame, int) or not 0 <= frame < target['num_frames']:
                issues.append({'video': video, 'issue': 'event_out_of_bounds', 'frame': frame})
            events.append((event.get('label'), frame))
        actual = Counter(events)
        wanted = Counter((e['label'], e['frame']) for e in target.get('events', []))
        matched_events += sum((actual & wanted).values())
        if actual != wanted:
            issues.append({'video': video, 'issue': 'event_mismatch',
                           'missing': list((wanted-actual).elements()),
                           'extra': list((actual-wanted).elements())})
        n = metadata.get('num_frames')
        if n is not None and n != target['num_frames']:
            issues.append({'video': video, 'issue': 'frame_count_mismatch',
                           'expected': target['num_frames'], 'actual': n})
    missing = sorted(set(expected) - set(seen))
    duplicates = {k: v for k, v in seen.items() if v > 1}
    return {'samples': len(records), 'expected_samples': len(reference),
            'missing_samples': missing, 'duplicate_samples': duplicates,
            'matched_events': matched_events, 'label_counts': dict(labels),
            'issues': issues, 'passed': not missing and not duplicates and not issues}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare(load_records(args.manifest), json.loads(args.reference.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in {'missing_samples', 'issues'}}))
    raise SystemExit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
