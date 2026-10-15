"""Raw DAT timing, parsing, graph and annotation contracts."""
import copy
import io
import json
import pickle
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from opensportslib.tracking import TracabTrackingReader
from opensportslib.tracking.tracab import parse_utc, utc_text
from opensportslib.datasets.utils.tracking import MISSING_VALUE, load_or_build_game_cache, parse_frame, slice_window
from opensportslib.core.utils.load_annotations import annotationstoe2eformat_tracking, load_annotations, select_annotation_head
from tests.helpers.tracab import write_tracab

pytestmark = pytest.mark.unit


def test_reader_timing_entities_gaps_and_worker_ownership(tmp_path):
    path = tmp_path / 'match.dat'
    descriptor = write_tracab(path)
    with TracabTrackingReader(path, descriptor) as reader:
        assert reader.source_frame_count == 9
        assert reader.playing_frame_count == 8
        row = reader.row_at(80)
        assert row['homePlayers'][0] == {'jerseyNum': 9, 'playerId': 'H9', 'x': 1, 'y': -2, 'speed': 1, 'trackId': '7', 'teamCode': '1', 'shirtNumber': '9'}
        assert len(row['awayPlayers']) == 1
        assert row['balls'][0]['z'] == .75
        assert row['balls'][0]['flags'] == ['SetHome']
        assert row['timestamp_utc'] == '2020-01-01 00:00:00.080000'
        for time in (0, 79, 240, 500, 799, 960, 1199, 1200):
            assert not reader.row_at(time)['homePlayers']
        assert reader.row_at(800)['sourceFrame'] == 200
        assert all(a < b for a, b in zip(reader.time_axis_ms, reader.time_axis_ms[1:]))
        with ThreadPoolExecutor(2) as pool:
            assert list(pool.map(lambda t: reader.row_at(t)['sourceFrame'], [80, 800] * 10)) == [100, 200] * 10
        with pickle.loads(pickle.dumps(reader)) as restored:
            assert restored.row_at(800) == reader.row_at(800)
    with TracabTrackingReader(io.BytesIO(path.read_bytes()), descriptor) as stream_reader:
        with pickle.loads(pickle.dumps(stream_reader)) as restored:
            assert restored.row_at(80)['homePlayers'][0]['playerId'] == 'H9'


@pytest.mark.parametrize('mutation,error', [
    ('reverse', 'non-monotonic'), ('duplicate', 'non-monotonic'),
    ('missing_boundary', 'boundaries'), ('nan', 'finite'),
    ('jersey', 'Unknown home jersey'), ('malformed', 'Malformed target'),
    ('ball', 'Malformed ball'), ('sections', 'expected frame'),
])
def test_invalid_records_fail_with_context(tmp_path, mutation, error):
    path = tmp_path / 'match.dat'
    descriptor = write_tracab(path)
    text = path.read_text()
    if mutation == 'reverse': text = '\n'.join(reversed(text.splitlines()))
    elif mutation == 'duplicate': text = text.splitlines()[0] + '\n' + text
    elif mutation == 'missing_boundary': text = '\n'.join(text.splitlines()[1:])
    elif mutation == 'nan': text = text.replace('-200', 'nan')
    elif mutation == 'jersey': text = text.replace('1,7,9,', '1,7,99,')
    elif mutation == 'malformed': text = text.replace('1,7,9,', '1,7,')
    elif mutation == 'ball': text = text.replace('H,Alive,SetHome', '')
    elif mutation == 'sections': text = text.replace(':', '|')
    path.write_text(text)
    with pytest.raises(ValueError, match=error): TracabTrackingReader(path, descriptor)


@pytest.mark.parametrize('mutation', ['missing_utc', 'fps', 'overlap', 'outside', 'teams', 'duplicate_jersey'])
def test_invalid_descriptors_fail(tmp_path, mutation):
    path = tmp_path / 'match.dat'
    d = write_tracab(path)
    if mutation == 'missing_utc': del d['UTC_time_start']
    elif mutation == 'fps': d['fps'] = 0
    elif mutation == 'overlap': d['metadata']['periods'][1]['start_utc'] = '2020-01-01 00:00:00.100000'
    elif mutation == 'outside': d['duration_ms'] = 100
    elif mutation == 'teams': d['metadata']['team_codes']['away'] = 1
    elif mutation == 'duplicate_jersey': d['metadata']['rosters']['home'] *= 2
    with pytest.raises(ValueError): TracabTrackingReader(path, d)


def test_graph_cache_matches_parquet_and_invalidates_alignment(tmp_path):
    path = tmp_path / 'match.dat'
    descriptor = write_tracab(path)
    features, times, positions = load_or_build_game_cache(str(path), descriptor=descriptor)
    assert features.shape == (30, 23, 8)
    assert features[10, 0, 0] == MISSING_VALUE
    with TracabTrackingReader(path, descriptor) as reader:
        rows = list(reader.iter_uniform_rows())
    for row in rows:
        for key in ('homePlayers', 'awayPlayers', 'balls'):
            row[key] = json.dumps(row[key])
    parquet = tmp_path / 'reference.parquet'
    pd.DataFrame(rows).to_parquet(parquet)
    other = load_or_build_game_cache(str(parquet))
    for a, b in zip((features, times, positions), other): np.testing.assert_array_equal(a, b)
    changed = copy.deepcopy(descriptor)
    changed['metadata']['periods'][1]['start_utc'] = '2020-01-01 00:00:00.840000'
    shifted = load_or_build_game_cache(str(path), descriptor=changed)[0]
    assert features[20, 0, 0] != MISSING_VALUE
    assert shifted[20, 0, 0] == MISSING_VALUE
    # Downsampling must not bridge missing native frames.
    dropped = np.array(features)
    dropped[3] = MISSING_VALUE
    window, _ = slice_window(dropped, positions, 1, 2, 2)
    assert window[-1, 0, 0] != MISSING_VALUE
    assert window[-1, 0, 5] == 0


def test_localization_selects_type_head_and_utc(tmp_path):
    path = tmp_path / 'match.dat'
    source = write_tracab(path)
    source['path'] = path.name
    document = {'labels': {'action': {'labels': ['pass']}, 'headers': {'labels': ['header']}},
                'data': [{'id': 'match', 'inputs': [{'type': 'video', 'path': 'missing.mp4'}, source],
                          'events': [{'head': 'action', 'label': 'pass', 'position_ms': 80},
                                     {'head': 'headers', 'label': 'header', 'position_ms': 0,
                                      'timestamp_utc': '2020-01-01 00:00:00.800000'}]}]}
    annotations = tmp_path / 'annotations.json'
    annotations.write_text(json.dumps(document))
    rows, head = annotationstoe2eformat_tracking(str(annotations), str(tmp_path), 25, 'tracking_tracab', 'headers')
    assert head == 'headers'
    assert rows[0]['events'] == [{'frame': 20, 'eval_frame': 20, 'label': 'header'}]
    assert rows[0]['input'] == source
    assert select_annotation_head(document) == 'action'
    document['labels'] = {'first': {}, 'second': {}}
    with pytest.raises(ValueError, match='annotation_head'): select_annotation_head(document)


def test_metadata_reference_resolution_and_cache_invalidation(tmp_path):
    path = tmp_path / 'tracking' / 'match.dat'
    descriptor = write_tracab(path, sidecar=True)
    metadata_path = path.with_suffix('.metadata.json')
    descriptor['metadata_path'] = 'tracking/match.metadata.json'
    before = copy.deepcopy(descriptor)
    with pytest.raises(ValueError, match='metadata_base_dir'):
        TracabTrackingReader(path, descriptor)
    with TracabTrackingReader(path, descriptor, metadata_base_dir=tmp_path) as reader:
        assert reader.fps == 25 and reader.pitch_length_m == 105
        assert reader.row_at(80)['homePlayers'][0]['playerId'] == 'H9'
    assert descriptor == before
    descriptor['metadata_path'] = str(metadata_path)
    load_or_build_game_cache(path, descriptor=descriptor)
    initial = set(tmp_path.rglob('*.features.npy'))
    metadata = json.loads(metadata_path.read_text())
    metadata['HomeTeam']['Players'][0]['PlayerID'] = 'replacement'
    metadata_path.write_text(json.dumps(metadata))
    load_or_build_game_cache(path, descriptor=descriptor)
    assert len(set(tmp_path.rglob('*.features.npy')) - initial) == 1
    with TracabTrackingReader(path, descriptor) as reader:
        assert reader.row_at(80)['homePlayers'][0]['playerId'] == 'replacement'


@pytest.mark.parametrize('mutation,error', [
    ('missing', 'metadata'), ('json', 'metadata'), ('roster', 'JerseyNo'),
    ('conflict', 'fps disagrees'), ('duplicate', 'Duplicate/invalid'),
])
def test_invalid_metadata_reference_has_context(tmp_path, mutation, error):
    path = tmp_path / 'match.dat'
    descriptor = write_tracab(path, sidecar=True)
    metadata_path = path.with_suffix('.metadata.json')
    metadata = json.loads(metadata_path.read_text())
    if mutation == 'missing': metadata_path.unlink()
    elif mutation == 'json': metadata_path.write_text('{broken')
    elif mutation == 'conflict': descriptor['fps'] = 30
    else:
        if mutation == 'roster': del metadata['HomeTeam']['Players'][0]['JerseyNo']
        else: metadata['HomeTeam']['Players'] *= 2
        metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=error) as caught:
        TracabTrackingReader(path, descriptor)
    assert str(metadata_path) in str(caught.value) or mutation == 'duplicate'


def test_native_records_preserve_tokens_coordinates_unknown_targets_and_extra_fields():
    from opensportslib.tracking import TracabDatReader
    raw = b'00100:1,007,09,-00125,0,0.00,extra;;3,12,0,5,6,7;-1,15,-1,0,0,0;:10,-20,0030,4,H,Alive,Unknown,;:\r\n'
    frame = next(iter(TracabDatReader(io.BytesIO(raw))))
    assert frame.raw == raw and frame.frame_id == '00100'
    assert len(frame.targets) == 3
    assert frame.targets[0].values == {'team_code': '1', 'tracking_id': '007', 'shirt_number': '09', 'x': '-00125', 'y': '0', 'speed': '0.00'}
    assert frame.targets[0].additional_fields == ('extra',)
    assert frame.balls[0].values['z'] == '0030'
    assert frame.balls[0].additional_fields == ('Unknown', '')
    empty = next(iter(TracabDatReader(io.BytesIO(b'101:::\n'))))
    assert empty.targets == empty.balls == []


def test_native_modes_keep_rejected_records_duplicates_and_gaps():
    from opensportslib.tracking import TracabDatReader
    raw = b'10:::\n10:::\n12:::\n11:::\nwrong|layout\n13:1,2,3,,0,1;::\n'
    frames = list(TracabDatReader(io.BytesIO(raw), mode='permissive'))
    assert b''.join(frame.raw for frame in frames) == raw
    assert [frame.rejected for frame in frames] == [False]*4 + [True]*2
    assert 'duplicate_frame' in [d.code for d in frames[1].diagnostics]
    assert 'frame_gap' in [d.code for d in frames[2].diagnostics]
    assert 'out_of_order' in [d.code for d in frames[3].diagnostics]
    with pytest.raises(ValueError, match='line 5'):
        list(TracabDatReader(io.BytesIO(raw)))
    assert len(list(TracabDatReader(io.BytesIO(raw.split(b'wrong')[0])))) == 4


def test_native_xml_preserves_namespaces_repetition_heatmaps_and_unknowns():
    from opensportslib.tracking import read_document, decode_heatmap
    raw = b'<?xml version="1.0"?><t:TracabFeed xmlns:t="urn:t" sTracabFeedId="TF05"><t:Team sHeatmap="0009"><t:Unknown x=""/><t:Unknown x="0"/><!--retain--></t:Team></t:TracabFeed>'
    document = read_document(io.BytesIO(raw), format='xml')
    assert document.raw == raw and document.feed_id == 'TF05'
    assert document.namespaces == [('t', 'urn:t')]
    team = list(document.value)[0]
    assert len(team) == 3 and [node.get('x') for node in list(team)[:2]] == ['', '0']
    assert list(document.heatmaps())[0]['encoded'] == '0009'
    with pytest.raises(ValueError, match='requires known'):
        decode_heatmap('0009', rows=2, columns=2, order='row-major', orientation=None, value_semantics=None)
    result = decode_heatmap('0012', rows=2, columns=2, order='column-major', orientation='fixture-defined', value_semantics='ordinal code')
    assert result['grid'] == [['0', '1'], ['0', '2']] and result['encoded'] == '0012'


def test_period_bounds_are_read_from_source_and_feeds_are_distinct(tmp_path):
    from pathlib import Path
    path = tmp_path / 'match.dat'
    descriptor = write_tracab(path, sidecar=True)
    assert 'start_frame' not in descriptor['metadata']['periods'][0]
    with TracabTrackingReader(path, descriptor) as reader:
        assert reader.periods[0][:2] == (100, 103)
        assert reader.native_frame(reader.time_axis_ms.index(80)).targets[0].fields[3] == '100'
        assert reader.documents['tf05_path'].feed_id == 'TF05'
        assert reader.documents['tf08_path'].feed_id == 'TF08'
        assert reader.transformations['speed'].startswith('unchanged')
    bad = copy.deepcopy(descriptor)
    bad['tf05_path'] = descriptor['tf08_path']
    with pytest.raises(ValueError, match='not interchangeable'):
        TracabTrackingReader(path, bad)
    metadata_path = Path(descriptor['metadata_path'])
    metadata = json.loads(metadata_path.read_text())
    metadata['GameID'] = '0001'
    metadata_path.write_text(json.dumps(metadata))
    Path(descriptor['tf08_path']).write_text('<TracabFeed sTracabFeedId="TF08"><TracabDocument iMatchId="2"/></TracabFeed>')
    with pytest.raises(ValueError, match='conflicts') as exc:
        TracabTrackingReader(path, descriptor)
    assert str(metadata_path) in str(exc.value) and descriptor['tf08_path'] in str(exc.value)


def test_identity_uses_team_and_active_interval_with_native_access(tmp_path):
    from pathlib import Path
    path = tmp_path / 'match.dat'
    descriptor = write_tracab(path, sidecar=True)
    metadata_path = Path(descriptor['metadata_path'])
    metadata = json.loads(metadata_path.read_text())
    metadata['HomeTeam']['TeamID'] = '001'
    metadata['HomeTeam']['Players'] = [
        {'PlayerID': '0009', 'JerseyNo': 9, 'StartFrameCount': 100, 'EndFrameCount': 103},
        {'PlayerID': '0099', 'JerseyNo': 9, 'StartFrameCount': 200, 'EndFrameCount': 203},
    ]
    metadata_path.write_text(json.dumps(metadata))
    with TracabTrackingReader(path, descriptor) as reader:
        assert reader.row_at(80)['homePlayers'][0]['playerId'] == '0009'
        assert reader.row_at(800)['homePlayers'][0]['playerId'] == '0099'
        assert reader.row_at(80)['homePlayers'][0]['teamId'] == '001'
