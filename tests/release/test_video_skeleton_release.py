"""Opt-in real-weight public tennis pose and full-pipeline verification."""
import json
import os
from pathlib import Path
import uuid

import numpy as np
import pytest

from tests.release._release_common import (
    DATA_DIR, OUTPUT_DIR, require_release_enabled, snapshot_dataset, record_release_metadata,
)

pytestmark = [pytest.mark.release, pytest.mark.gpu, pytest.mark.network, pytest.mark.pretrained]


def test_public_tennis_video_skeleton_gpu():
    require_release_enabled()
    if os.environ.get('OSL_RELEASE_POSE') != '1':
        pytest.skip('Pose release verification is additionally opt-in with OSL_RELEASE_POSE=1')
    import torch
    assert torch.cuda.is_available(), 'Pose release verification requires working CUDA'
    manifest_path = os.environ.get('OSL_RELEASE_POSE_MANIFEST')
    assert manifest_path, 'Supply a rights-reviewed public-video manifest via OSL_RELEASE_POSE_MANIFEST'
    manifest = json.loads(Path(manifest_path).read_text())
    assert len(manifest['revision']) == 40, 'Pin the public dataset commit'
    assert manifest.get('rights'), 'Document public-video usage rights'
    cases = {tag for video in manifest['videos'] for tag in video['cases']}
    assert {'far_player','fast_swing','occlusion','camera_cut','extra_people'} <= cases
    root = snapshot_dataset(manifest['repo_id'], DATA_DIR/'pose-public',
                            allow_patterns=[v['path'] for v in manifest['videos']],
                            revision=manifest['revision'])
    from opensportslib.core.config.editable import Config
    from opensportslib.core.config.runtime_adapter import dict_to_namespace
    from opensportslib.preprocessing.config import preprocessing_config
    from opensportslib.preprocessing import extract, iter_skeletons, SkeletonCache, player_features
    from opensportslib.preprocessing.benchmark import benchmark
    config = dict_to_namespace(Config.from_file(os.environ.get(
        'OSL_RELEASE_POSE_CONFIG','examples/tennis/localization_skeleton.yaml')).get_config())
    settings = preprocessing_config(config)
    output = OUTPUT_DIR/'video-skeleton'/uuid.uuid4().hex
    output.mkdir(parents=True)
    record_release_metadata('pose_public_dataset',repo_id=manifest['repo_id'],
                            revision=manifest['revision'],rights=manifest['rights'],output=str(output))
    for index,entry in enumerate(manifest['videos']):
        video = root/entry['path']
        cache_path = output/f'{index}-cache'
        extract(video,cache_path,settings)
        for live,cached in zip(iter_skeletons(video,settings),SkeletonCache(cache_path),strict=True):
            np.testing.assert_array_equal(player_features(live),player_features(cached))
            assert live.timestamp==cached.timestamp and live.shot_id==cached.shot_id
            np.testing.assert_array_equal(live.track_ids,cached.track_ids)
        report=benchmark(config,video,output/f'{index}-benchmark.json',repeats=2)
        assert all(run['source_fps_processed']>0 for run in report['runs'])
        assert not any(run['memory_sampling_error'] for run in report['runs'])
        record_release_metadata('pose_benchmark',video=entry['path'],cases=entry['cases'],
                                provenance=report['provenance'],report=str(output/f'{index}-benchmark.json'))
