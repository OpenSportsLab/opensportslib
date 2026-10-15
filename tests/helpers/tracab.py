"""Small synthetic two-period TRACAB fixture shared by reader and loader tests."""
import json


def write_tracab(path, *, sidecar=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = [100, 101, 102, 103, 150, 200, 201, 202, 203]
    path.write_text(''.join(
        f'{frame}:1,7,9,{frame},-200,1;0,4,3,-300,400,2;3,8,0,0,0,0;:-100,200,75,3,H,Alive,SetHome;:\n'
        for frame in frames
    ))
    descriptor = {
        'type': 'tracking_tracab', 'path': str(path), 'fps': 25,
        'UTC_time_start': '2020-01-01 00:00:00.000000', 'duration_ms': 1200,
        'metadata': {
            'team_codes': {'home': 1, 'away': 0},
            'pitch_length_m': 105, 'pitch_width_m': 68,
            'rosters': {'home': [{'jerseyNum': 9, 'playerId': 'H9'}],
                        'away': [{'jerseyNum': 3, 'playerId': 'A3'}]},
            'periods': [
                {'start_frame': 100, 'end_frame': 103, 'start_utc': '2020-01-01 00:00:00.080000'},
                {'start_frame': 200, 'end_frame': 203, 'start_utc': '2020-01-01 00:00:00.800000'},
            ],
        },
    }
    if sidecar:
        tracab = descriptor['metadata']
        metadata = {
            'FrameRate': descriptor.pop('fps'),
            'PitchLongSide': tracab.pop('pitch_length_m') * 100,
            'PitchShortSide': tracab.pop('pitch_width_m') * 100,
        }
        for number, period in enumerate(tracab['periods'], 1):
            metadata[f'Phase{number}StartFrame'] = period.pop('start_frame')
            metadata[f'Phase{number}EndFrame'] = period.pop('end_frame')
            period['period'] = number
        for side, roster in tracab.pop('rosters').items():
            metadata[side.title() + 'Team'] = {'Players': [
                {'JerseyNo': p['jerseyNum'], 'PlayerID': p['playerId']} for p in roster]}
        metadata_path = path.with_suffix('.metadata.json')
        metadata_path.write_text(json.dumps(metadata))
        descriptor['metadata_path'] = str(metadata_path)
        for feed in ('TF05', 'TF08'):
            xml = path.with_suffix('.' + feed + '.xml')
            xml.write_text(f'<TracabFeed sVersion="1.0" sTracabFeedId="{feed}"><TracabDocument><Unknown value="0001"/></TracabDocument></TracabFeed>')
            descriptor[feed.lower() + '_path'] = str(xml)
    return descriptor
