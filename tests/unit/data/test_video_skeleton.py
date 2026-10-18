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


def test_cached_spotting_dataset_labels_padding_and_collation(tmp_path):
    import torch
    from opensportslib.core.config.editable import Config
    from opensportslib.core.config.runtime_adapter import dict_to_namespace
    from opensportslib.datasets.skeleton import SkeletonSpottingDataset
    frames=list(observations(9,shot_at=5))
    write_cache(tmp_path/'pose',frames,{'settings':{'sample_fps':25}},3)
    path=tmp_path/'labels.json'
    path.write_text(json.dumps({'data':[{'id':'match','inputs':[{'type':'skeleton_cache','path':'pose'}],
                                      'events':[{'label':'swing','position_ms':80}]}]}))
    payload=Config.from_file('examples/tennis/localization_skeleton.yaml').get_config()
    payload['DATA']['common']['classes']=['swing']
    payload['DATA']['inputs']['video']['sampling']['clip_len']=4
    config=dict_to_namespace(payload)
    dataset=SkeletonSpottingDataset(config,path,tmp_path,'train',2)
    assert dataset.get_labels('match')[2]==1
    assert len(dataset)==3
    tail=dataset[1]
    assert tail['time_mask'].tolist()==[True,True,True,False]
    batch=dataset.collate_fn([tail])
    assert batch['time_mask'].tolist()==[[True,True,True,False]]
    assert batch['frame'].num_graphs==4
    assert batch['frame'].node_mask.dtype==torch.bool
    test=SkeletonSpottingDataset(config,path,tmp_path,'test',2)
    result=test.collate_fn([test[1]])
    assert result['frame'][0].time_mask.sum()==3
    payload['DATA']['inputs']['video']['sampling']['extract_fps']=20
    with pytest.raises(ValueError,match='sample rate'):
        SkeletonSpottingDataset(dict_to_namespace(payload),path,tmp_path,'test',2)


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
