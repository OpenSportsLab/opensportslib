"""Canonical pose/session/cache contracts, without optional pose runtimes."""
from dataclasses import replace
from fractions import Fraction
import json

import numpy as np
import pytest

from opensportslib.preprocessing import SkeletonFrame, SkeletonSession, SkeletonCache, player_features
from opensportslib.preprocessing.cache import write_cache, cache_identity
from opensportslib.preprocessing.windows import iter_windows, RollingScores

pytestmark = pytest.mark.unit


class Backend:
    provenance = {'fake_test_backend': True}

    def __init__(self):
        self.calls = 0
        self.absent = False

    def predict(self, image):
        self.calls += 1
        if self.absent:
            return np.empty((0,4)), np.empty((0,17,2)), np.empty((0,17))
        return np.array([[10,10,30,60]], np.float32), np.full((1,17,2),20,np.float32), np.ones((1,17),np.float32)


def observations(count=10, shot_at=None):
    backend = Backend()
    session = SkeletonSession(backend, {'selection':'all_people','sample_fps':25})
    for i in range(count):
        image = np.full((80,100,3), 255 if shot_at is not None and i >= shot_at else 0, np.uint8)
        yield from session.push(image, timestamp=i/25, frame_index=i, pts=i,
                                time_base=(1,25))
    session.finish()
    session.close()


def test_feature_conventions_masks_and_confidence():
    frame = next(observations())
    frame.joint_valid[0,0] = False
    frame.confidence[0,0] = .1
    features = player_features(frame)
    assert features.shape == (2,73)
    np.testing.assert_allclose(features[0,:4], [.2,.4375,.2,.625])
    np.testing.assert_allclose(features[0,4:8], [0,0,.1,0])
    assert not features[1].any()
    assert frame.track_ids.tolist() == [0,-1]
    with pytest.raises(ValueError, match='finite'):
        player_features(replace(frame, timestamp=float('nan')))


def test_sampling_occlusion_retention_cut_and_session_isolation():
    backend = Backend()
    session = SkeletonSession(backend, {'selection':'all_people','sample_fps':25})
    black = np.zeros((80,100,3),np.uint8)
    def push(t, image=black):
        return session.push(image,timestamp=t,frame_index=round(t*100),pts=round(t*100),time_base=(1,100))
    assert push(0)[0].track_ids[0] == 0
    assert push(.01) == []
    backend.absent = True
    assert not push(.04)[0].player_valid.any()
    backend.absent = False
    assert push(.08)[0].track_ids[0] == 0
    assert push(.8)[0].track_ids[0] == 1
    assert push(.84,255-black)[0].shot_id == 1
    assert session.association.tracks[0].identity == 2
    with pytest.raises(ValueError,match='increase'):
        push(.84)
    session.finish()
    with pytest.raises(RuntimeError,match='finished'):
        push(.9)
    assert next(observations()).track_ids[0] == 0


def test_chunked_cache_equivalence_invalidation_and_cleanup(tmp_path):
    source = tmp_path/'video'; source.write_bytes(b'test-video')
    frames = list(observations(19,shot_at=9))
    identity = cache_identity(source, {'sample_fps':25}, Backend.provenance)
    for size in (1,7,256):
        path = write_cache(tmp_path/str(size), iter(frames), identity, size)
        cache = SkeletonCache(path,identity)
        assert len(cache) == len(frames)
        for live,cached in zip(frames,cache):
            np.testing.assert_array_equal(player_features(live),player_features(cached))
            assert (live.pts,live.timestamp,live.shot_id)==(cached.pts,cached.timestamp,cached.shot_id)
        with pytest.raises(ValueError,match='mismatch'):
            SkeletonCache(path,{**identity,'video_sha256':'different'})
        with pytest.raises(FileExistsError):
            write_cache(path,frames,identity)
    def broken():
        yield frames[0]
        raise RuntimeError('interrupted')
    with pytest.raises(RuntimeError,match='interrupted'):
        write_cache(tmp_path/'incomplete',broken(),identity,1)
    assert not (tmp_path/'incomplete').exists()
    assert not list(tmp_path.glob('.skeleton-*'))
    with pytest.raises(ValueError,match='Incomplete'):
        SkeletonCache(tmp_path/'incomplete')


@pytest.mark.parametrize('count',[1,4,5,8,9,30])
def test_rolling_equals_offline_and_shot_boundaries(count):
    windows = list(iter_windows(observations(count,shot_at=7),4,2))
    boundaries = list(iter_windows(observations(count,shot_at=7),4,2,include_boundaries=True))
    assert sum(w is None for w in boundaries) == int(count > 7)
    sums,counts = np.zeros((count,3)),np.zeros(count)
    rolling = RollingScores(); results=[]
    for number,window in enumerate(windows):
        real = window.frames[:len(window.indices)]
        assert len({f.shot_id for f in real}) == 1
        results.extend(rolling.finalize_before(window.indices[0]))
        scores = np.full((4,3),number+.25)
        rolling.add(window,scores)
        for offset,index in enumerate(window.indices):
            sums[index]+=scores[offset];counts[index]+=1
        assert len(rolling.pending)<=4
    results.extend(rolling.finish())
    assert [row[0] for row in results]==list(range(count))
    np.testing.assert_allclose(np.stack([row[2] for row in results]),sums/counts[:,None])


def test_decode_timestamp_and_live_cache_alignment(tmp_path):
    import av
    from opensportslib.preprocessing.session import extract, iter_skeletons, decode_video
    path = tmp_path/'clip.mp4'
    with av.open(str(path),'w') as out:
        stream=out.add_stream('mpeg4',rate=Fraction(30000,1001));stream.width=100;stream.height=80
        for i in range(12):
            f=av.VideoFrame.from_ndarray(np.zeros((80,100,3),np.uint8),format='bgr24')
            for packet in stream.encode(f):out.mux(packet)
        for packet in stream.encode():out.mux(packet)
    settings={'selection':'all_people','sample_fps':25}
    live=list(iter_skeletons(path,settings,Backend()))
    extract(path,tmp_path/'cache',settings,Backend(),chunk_size=3)
    read=list(iter_skeletons(path,{**settings,'cache':{'mode':'read','path':str(tmp_path/'cache')}}))
    assert len(live)==len(read)>1
    assert len(list(decode_video(path)))==12
    for a,b in zip(live,read):
        np.testing.assert_array_equal(player_features(a),player_features(b))
        assert a.timestamp==b.timestamp


def test_conservative_court_selection_and_ambiguity():
    import cv2
    from opensportslib.preprocessing.selection import TennisSelector
    court=np.zeros((180,320,3),np.uint8)
    for a,b in [((120,40),(60,165)),((200,40),(260,165)),
                ((120,40),(200,40)),((60,165),(260,165)),((90,103),(230,103))]:
        cv2.line(court,a,b,(255,255,255),3)
    boxes=np.array([[145,12,175,43],[140,110,180,162]],np.float32)
    selector=TennisSelector()
    assert selector.select(court,boxes)==([],False)
    selected,valid=selector.select(court,boxes)
    assert valid and selected==[(0,0),(1,1)]
    _,valid=selector.select(np.zeros_like(court),boxes)
    assert not valid


def test_configuration_errors_and_optional_runtime_isolation():
    from opensportslib.preprocessing.config import validate_preprocessing
    for cfg,match in [({'max_players':0},'max_players'),({'max_candidates':1},'max_candidates'),
                      ({'sample_fps':0},'sample_fps'),({'cache':{'mode':'write'}},'extract'),
                      ({'backend':'unknown'},'backend')]:
        with pytest.raises(ValueError,match=match):validate_preprocessing(cfg)


def test_top_down_filters_court_people_before_pose_and_candidate_limit():
    import cv2
    image=np.zeros((180,320,3),np.uint8)
    for a,b in [((120,40),(60,165)),((200,40),(260,165)),
                ((120,40),(200,40)),((60,165),(260,165)),((90,103),(230,103))]:
        cv2.line(image,a,b,(255,255,255),3)
    class TopDown:
        one_stage=False
        def __init__(self):self.crops=[]
        def detect(self,image):
            return np.array([[0,0,40,50]]*10+[[145,12,175,43],[140,110,180,162]],np.float32)
        def estimate(self,image,boxes):
            self.crops.append(len(boxes))
            return np.full((len(boxes),17,2),20,np.float32),np.ones((len(boxes),17),np.float32)
    backend=TopDown()
    session=SkeletonSession(backend,{'sample_fps':25,'max_candidates':8})
    for index in range(2):
        out=session.push(image,timestamp=index/25,frame_index=index,pts=index,time_base=(1,25))
    assert backend.crops==[0,2]
    assert out[0].player_valid.tolist()==[True,True]


def test_weight_checksum_and_no_detection_backend_contract(tmp_path):
    from opensportslib.preprocessing.backend import resolve_artifact, RTMLibBackend
    from opensportslib.preprocessing.cache import file_sha256
    model=tmp_path/'model.onnx';model.write_bytes(b'fake external model')
    cfg={'artifacts':{'rtmpose_s':{'path':str(model),'sha256':file_sha256(model)}}}
    path,size,provenance=resolve_artifact('rtmpose_s',cfg)
    assert size==(192,256) and provenance['sha256']==file_sha256(path)
    cfg['artifacts']['rtmpose_s']['sha256']='incorrect'
    with pytest.raises(ValueError,match='checksum'):
        resolve_artifact('rtmpose_s',cfg)
    backend=RTMLibBackend.__new__(RTMLibBackend)
    backend.one_stage=False;backend.max_candidates=8
    backend.timings={'detection_seconds':0.,'pose_seconds':0.}
    backend.detector=lambda image:np.empty((0,4))
    def unexpected_pose(*args,**kwargs):
        pytest.fail('No detections must never invoke whole-image RTMPose fallback')
    backend.pose=unexpected_pose
    boxes,joints,scores=backend.predict(np.zeros((80,100,3),np.uint8))
    assert boxes.shape==(0,4) and joints.shape==(0,17,2) and scores.shape==(0,17)


def test_raw_clip_sampling_geometry_padding_and_multiview_collation(tmp_path):
    import torch
    from tests.helpers.video_pose import write_video
    from opensportslib.datasets.video_raw import read_raw_clip, uniform_indices, video_collate
    path = write_video(tmp_path/'clip.mp4', 3)
    assert uniform_indices(0, 3, 5) == [0, 1, 1, 2, 2]
    frames, meta = read_raw_clip(path, [-1, 0, 1, 1, 2, -1], resize={'width':50,'height':40})
    assert frames.dtype == torch.uint8 and frames.shape == (6,3,40,50)
    assert [f['valid'] for f in meta['frames']] == [False,True,True,True,True,False]
    assert meta['frames'][1]['image_size'] == [100,80]
    assert meta['frames'][2]['timestamp'] == meta['frames'][3]['timestamp']
    larger, other = read_raw_clip(path, [0, 2])
    batch = video_collate([{'pixel_values':[frames, larger], 'video_meta':[meta, other], 'id':'a'},
                           {'pixel_values':[larger], 'video_meta':[other], 'id':'b'}])
    assert batch['pixel_values'].shape == (2,2,3,6,80,100)
    assert batch['view_mask'].tolist() == [[True,True],[True,False]]
    assert not batch['time_mask'][0,0,0]


def test_model_frontend_repeats_padding_cuts_sessions_and_resize(tmp_path):
    import torch
    from tests.helpers.video_pose import write_video, inject_backend
    from opensportslib.datasets.video_raw import read_raw_clip
    from opensportslib.models.preprocessors import PoseTracking
    path = write_video(tmp_path/'clip.mp4', 8, cut=4)
    frames, meta = read_raw_clip(path, [-1,0,1,1,2,3,4,5,6,7], resize={'width':50,'height':40})
    frontend=PoseTracking({'selection':'all_people'})
    backend=inject_backend(frontend)
    rows=frontend.observations(frames,meta)
    assert backend.calls==8
    assert rows[0].frame_index==-1 and not rows[0].player_valid.any()
    assert rows[2].track_ids.tolist()==rows[3].track_ids.tolist()
    np.testing.assert_array_equal(rows[1].boxes[0],[20,20,60,120])
    assert rows[-1].image_size.tolist()==[100,80]
    assert rows[5].shot_id==0 and rows[6].shot_id==1
    session=frontend.open_session()
    try:
        chunked=[]
        for a,b in [(0,3),(3,6),(6,10)]:
            chunked.extend(session.push(frames[a:b],{**meta,'frames':meta['frames'][a:b]}))
        for a,b in zip(rows,chunked):
            np.testing.assert_array_equal(player_features(a),player_features(b))
            np.testing.assert_array_equal(a.track_ids,b.track_ids)
        assert len(session.recent)<=8
        session.finish()
        with pytest.raises(RuntimeError,match='finished'):
            session.push(frames,meta)
    finally:session.close()
    with pytest.raises(RuntimeError,match='closed'):session.push(frames,meta)
    frontend.train()
    assert not frontend.training and not list(frontend.parameters()) and not frontend.state_dict()


def test_exact_clip_cache_matches_live_without_runtime_or_state_leakage(tmp_path):
    from tests.helpers.video_pose import write_video, inject_backend
    from opensportslib.datasets.video_raw import read_raw_clip
    from opensportslib.models.preprocessors import PoseTracking
    path=write_video(tmp_path/'clip.mp4',5)
    settings={'selection':'all_people','cache':{'mode':'read_write','path':str(tmp_path/'cache')},
              'artifacts':{'yolox_m':{'sha256':'detector-test','path':str(tmp_path/'unavailable-detector.onnx')},
                           'rtmpose_s':{'sha256':'pose-test','path':str(tmp_path/'unavailable-pose.onnx')}}}
    frontend=PoseTracking(settings);backend=inject_backend(frontend)
    frames,meta=read_raw_clip(path,[0,1,2])
    live=frontend.observations(frames,meta)
    assert backend.calls==3
    cached=frontend.observations(frames,meta)
    assert backend.calls==3
    reader=PoseTracking({**settings,'cache':{'mode':'read','path':str(tmp_path/'cache')}})
    cached=reader.observations(frames,meta)
    assert not reader._backends
    for a,b in zip(live,cached):np.testing.assert_array_equal(player_features(a),player_features(b))
    frames,meta=read_raw_clip(path,[1,2,3])
    with pytest.raises(FileNotFoundError,match='Exact-clip'):reader.observations(frames,meta)
    assert frontend.observations(frames,meta)[0].track_ids[0]==0
    assert len(list((tmp_path/'cache').iterdir()))==2


def test_pose_observer_exports_live_and_cache_hits_without_leaking_context(tmp_path):
    from tests.helpers.video_pose import write_video, inject_backend
    from opensportslib.datasets.video_raw import read_raw_clip
    from opensportslib.models.preprocessors import PoseTracking
    from opensportslib.preprocessing.diagnostics import PoseTraceWriter
    path = write_video(tmp_path / 'clip.mp4', 3)
    frontend = PoseTracking({'selection': 'all_people', 'cache': {'mode': 'read_write', 'path': str(tmp_path / 'cache')}})
    backend = inject_backend(frontend)
    frames, meta = read_raw_clip(path, [-1, 0, 0, 2])
    captured = []
    with PoseTraceWriter(tmp_path / 'poses.jsonl') as trace, frontend.observe(trace):
        live = frontend.observations(frames, meta)
        cached = frontend.observations(frames, meta)
        with pytest.raises(RuntimeError, match='observer scope'):
            with frontend.observe(lambda rows, metadata: captured.append(rows)):
                raise RuntimeError('observer scope')
        assert trace.requests == 2 and trace.observations == 8
    assert backend.calls == 2 and captured == []
    # Closed diagnostic observers are not invoked by later forwards.
    frontend.observations(frames, meta)
    exported = [json.loads(line) for line in (tmp_path / 'poses.jsonl').read_text().splitlines()]
    assert len(exported) == 8 and [row['request_index'] for row in exported] == [0] * 4 + [1] * 4
    for row, frame in zip(exported, live + cached):
        np.testing.assert_array_equal(row['player_features'], player_features(frame))
        assert row['skeleton']['frame_index'] == frame.frame_index
    other = PoseTracking({'selection': 'all_people'})
    inject_backend(other)
    with frontend.observe(lambda rows, metadata: captured.append(rows)):
        other.observations(frames, meta)
    assert captured == []


def test_prediction_overlay_preserves_vfr_unsampled_frames_and_masks(tmp_path, monkeypatch):
    import av
    import cv2
    from tests.helpers.video_pose import write_video, inject_backend
    from opensportslib.datasets.video_raw import read_raw_clip
    from opensportslib.models.preprocessors import PoseTracking
    from opensportslib.preprocessing.diagnostics import PoseTraceWriter
    from opensportslib.preprocessing.visualization import render_prediction_overlay
    path = write_video(tmp_path / 'vfr.mp4', 4, frame_indices=[3, 4, 6, 9])
    frontend = PoseTracking({'selection': 'all_people'})
    backend = inject_backend(frontend)
    trace_path = tmp_path / 'poses.jsonl'
    with PoseTraceWriter(trace_path) as trace, frontend.observe(trace):
        frontend.observations(*read_raw_clip(path, [0]))
        backend.absent = True
        frontend.observations(*read_raw_clip(path, [2]))
    boxes = []
    original = cv2.rectangle
    def record_box(image, start, end, *args, **kwargs):
        boxes.append((start, end))
        return original(image, start, end, *args, **kwargs)
    monkeypatch.setattr(cv2, 'rectangle', record_box)
    predictions = {'task': 'action_classification', 'data': [{'labels': {'action': {'label': 'swing', 'confidence': .9}}}]}
    output = tmp_path / 'annotated.mp4'
    result = render_prediction_overlay(path, trace_path, predictions, output)
    assert result['source_frames'] == 4 and result['sampled_source_frames'] == 2
    assert boxes == [((10, 10), (30, 60))]
    with av.open(str(path)) as movie:
        decoded = list(movie.decode(video=0))
        source = [float(frame.pts * frame.time_base) for frame in decoded]
        source_durations = [float(frame.duration * frame.time_base) for frame in decoded]
        source_duration = float(movie.streams.video[0].duration * movie.streams.video[0].time_base)
    with av.open(str(output)) as movie:
        decoded = list(movie.decode(video=0))
        rendered = [float(frame.pts * frame.time_base) for frame in decoded]
        rendered_durations = [float(frame.duration * frame.time_base) for frame in decoded]
        rendered_duration = float(movie.streams.video[0].duration * movie.streams.video[0].time_base)
    np.testing.assert_allclose(rendered, np.array(source) - source[0], atol=1e-6)
    np.testing.assert_allclose(rendered_durations, source_durations, atol=1e-6)
    assert abs(rendered_duration - source_duration) < 1e-6
    with pytest.raises(FileExistsError):
        render_prediction_overlay(path, trace_path, predictions, output)
    rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
    rows[0]['skeleton']['pts'] += 1
    trace_path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(ValueError, match='timestamp mismatch'):
        render_prediction_overlay(path, trace_path, predictions, tmp_path / 'bad.mp4')
    assert not (tmp_path / 'bad.mp4').exists() and not (tmp_path / 'bad.part.mp4').exists()


def test_pose_config_contract_and_legacy_migration():
    from opensportslib.core.config.editable import Config
    from opensportslib.core.config.validate import validate_config
    from opensportslib.preprocessing.config import requires_raw_video
    for task in ['classification','localization']:
        for suffix in ['', '_dali']:
            payload=Config.from_file(f'examples/tennis/{task}_skeleton{suffix}.yaml').get_config()
            validate_config(payload)
            assert requires_raw_video(payload)
            assert payload['DATA']['inputs']['video']['representation']=='raw'
        payload['DATA']['inputs']['video']['representation']='skeleton_graph'
        with pytest.raises(ValueError,match='BREAKING CHANGE'):validate_config(payload)


def test_dali_adapter_preserves_sampler_requests_and_labels(tmp_path,monkeypatch):
    import torch
    from tests.helpers.video_pose import pose_config
    from opensportslib.core.config.runtime_adapter import dict_to_namespace
    from opensportslib.datasets.classification_dataset import VideoDataset
    from opensportslib.datasets.video_raw import build_video_loader, read_raw_clip
    payload=pose_config('classification',tmp_path)
    payload['DATA']['common']['runtime']['loader_backend']='dali'
    dataset=VideoDataset(dict_to_namespace(payload),str(tmp_path/'labels.json'),None)
    calls=[]
    def fake_decode(path,indices,clocks,info,device_id):
        calls.append(indices)
        frames,_=read_raw_clip(path,indices)
        return {i: f.permute(1,2,0).numpy() for i,f in zip(indices,frames)}
    monkeypatch.setattr('opensportslib.datasets.video_dali.decode_selected',fake_decode)
    sampler=torch.utils.data.WeightedRandomSampler([1.],3,replacement=True)
    loader=build_video_loader(dataset,batch_size=2,sampler=sampler,num_workers=4,drop_last=False)
    batches=list(loader)
    assert loader.sampler is sampler and loader.num_workers==0
    assert [len(b['id']) for b in batches]==[2,1] and len(calls)==3
    assert all(b['labels'].tolist()==[0]*len(b['id']) for b in batches)


def test_irregular_source_timestamps_are_preserved(tmp_path):
    from tests.helpers.video_pose import write_video
    from opensportslib.datasets.video_raw import read_raw_clip
    path=write_video(tmp_path/'vfr.mp4',4,frame_indices=[0,1,3,6])
    _,meta=read_raw_clip(path,[0,1,2,3])
    np.testing.assert_allclose([f['timestamp'] for f in meta['frames']],[0,.04,.12,.24],atol=1e-6)
    assert len({tuple(f['time_base']) for f in meta['frames']})==1


def test_stream_chunk_sizes_and_early_close_preserve_session_contract(tmp_path,monkeypatch):
    from tests.helpers.video_pose import pose_config,write_video,inject_backend
    from opensportslib.core.config.runtime_adapter import dict_to_namespace
    from opensportslib.models.preprocessors import PoseTracking
    from opensportslib.datasets.video_raw import read_raw_clip
    config=dict_to_namespace(pose_config('localization',tmp_path))
    path=write_video(tmp_path/'clip.mp4',9,cut=4)
    frontend=PoseTracking({'selection':'all_people'})
    inject_backend(frontend)
    frames,meta=read_raw_clip(path,list(range(9)))
    expected=frontend.observations(frames,meta)
    for size in (1,2,4,7):
        config.DATA.inputs.video.sampling.clip_len=size
        actual=list(frontend.iter_video(path,config))
        assert len(actual)==9
        for a,b in zip(expected,actual):
            np.testing.assert_array_equal(player_features(a),player_features(b))
            np.testing.assert_array_equal(a.track_ids,b.track_ids)
            assert (a.timestamp,a.pts,a.shot_id)==(b.timestamp,b.pts,b.shot_id)
    sessions=[]
    original=frontend.open_session
    def capture():
        session=original();sessions.append(session);return session
    monkeypatch.setattr(frontend,'open_session',capture)
    one=frontend.iter_video(path,config);two=frontend.iter_video(path,config)
    assert next(one).track_ids[0]==next(two).track_ids[0]==0
    one.close();two.close()
    assert all(s.closed and not s.recent for s in sessions)


def test_cache_warming_uses_existing_video_loader(tmp_path,monkeypatch):
    from tests.helpers.video_pose import pose_config,PoseBackend
    from opensportslib.core.config.runtime_adapter import dict_to_namespace
    from opensportslib.preprocessing.cache import warm_clip_cache
    monkeypatch.setattr('opensportslib.preprocessing.backend.RTMLibBackend',PoseBackend)
    config=dict_to_namespace(pose_config('classification',tmp_path))
    cache=tmp_path/'warmed'
    assert warm_clip_cache(config,cache,max_clips=1)==1
    entries=list(cache.iterdir())
    assert len(entries)==1
    reader=SkeletonCache(entries[0])
    assert reader.manifest['identity']['reset_policy']=='independent_clip_v1'
    assert len(reader)==4
    assert json.loads((tmp_path/'labels.json').read_text())['data'][0]['inputs'][0]['type']=='video'
