from pathlib import Path
import json

import pandas as pd
import pytest

from opensportslib.tools import convert_json_to_parquet, convert_parquet_to_json, parse_shard_size


def test_finediving_preparation_preserves_pixels_frames_and_resumes(tmp_path):
    import importlib.util
    import io
    import zipfile
    import numpy as np
    from PIL import Image

    spec = importlib.util.spec_from_file_location(
        'finediving_preparation', Path(__file__).parents[3] / 'tools/convert/prepare_finediving_spotting.py')
    preparation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preparation)
    ref = tmp_path / 'reference'
    ref.mkdir()
    (ref / 'class.txt').write_text('Entry\n')
    archive = tmp_path / 'frames.zip'
    with zipfile.ZipFile(archive, 'w') as source:
        for index, split in enumerate(('train', 'val', 'test'), 1):
            identity = f'01__{index}'
            row = {'video': identity, 'num_frames': 3, 'fps': -1,
                   'events': [{'frame': 1, 'label': 'Entry'}]}
            (ref / f'{split}.json').write_text(json.dumps([row]))
            for frame in range(3):
                pixels = np.random.default_rng(index*10+frame).integers(0, 256, (16, 20, 3), dtype=np.uint8)
                data = io.BytesIO()
                Image.fromarray(pixels).save(data, format='JPEG')
                source.writestr(f'FINADiving_MTL_256s/01/{index}/{100+frame:08d}.jpg', data.getvalue())
    output = tmp_path / 'prepared'
    report = preparation.prepare(archive, ref, output, workers=2, source_commit='pinned-test')
    assert report['clips'] == 3 and report['frames'] == 9
    doc = json.loads((output / 'train.json').read_text())
    record = doc['data'][0]
    assert record['events'][0]['frame'] == 1
    assert record['events'][0]['position_ms'] == 40
    assert record['metadata']['source_fps'] is None
    assert record['metadata']['source_frame_paths'][0].endswith('00000100.jpg')
    video = output / record['inputs'][0]['path']
    original_hash = preparation.checksum(video)
    video.write_bytes(b'corrupt')
    preparation.prepare(archive, ref, output, workers=1, source_commit='pinned-test')
    receipt = json.loads((output / 'receipts/01__1.json').read_text())
    assert preparation.checksum(video) == receipt['sha256']
    assert preparation.checksum(video) != preparation.hashlib.sha256(b'corrupt').hexdigest()
    assert len(original_hash) == 64
    with pytest.raises(ValueError, match='checksum mismatch'):
        preparation.prepare(archive, ref, output, source_commit='test', expected_archive_sha256='wrong')


def test_spotting_audit_uses_exact_frames_and_reports_missing_samples():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'spotting_audit', Path(__file__).parents[3] / 'tools/convert/audit_spotting_dataset.py')
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    ref = [{'video': 'clip', 'fps': 30000/1001, 'num_frames': 40,
            'events': [{'frame': 1, 'label': 'hit'}]},
           {'video': 'missing', 'fps': 25, 'num_frames': 2, 'events': []}]
    records = [{'inputs': [{'path': 'clip.mp4'}],
                'events': [{'position_ms': 33, 'label': 'hit'}]}]
    result = audit.compare(records, ref)
    assert result['matched_events'] == 1
    assert result['missing_samples'] == ['missing']
    assert result['issues'] == [] and not result['passed']
    records[0]['events'][0]['frame'] = 2
    assert any(x['issue'] == 'event_mismatch' for x in audit.compare(records, ref)['issues'])


def test_tennis_cleanup_repairs_boundaries_without_changing_labels(tmp_path):
    import importlib.util
    import numpy as np
    import av
    spec = importlib.util.spec_from_file_location(
        'tennis_cleanup', Path(__file__).parents[3] / 'tools/convert/clean_tennis_spotting.py')
    cleanup = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cleanup)
    source, reference, repairs = (tmp_path/name for name in ['source','reference','repairs'])
    reference.mkdir()
    repairs.mkdir()
    pixels = [np.full((16,20,3),i*30,dtype=np.uint8) for i in range(3)]
    for split,ref_split,n in [('train','train',3),('validation','val',1),('test','test',2)]:
        identity = f'clip_{split}'
        video = source/split/'media'/f'{identity}.mkv'
        cleanup.write_lossless_frames(iter(pixels[:n]),video,25)
        ref = {'video':identity,'num_frames':2,'fps':25,
               'events':[{'label':'hit','frame':1}]}
        (reference/f'{ref_split}.json').write_text(json.dumps([ref]))
        row = {'id':identity,'inputs':[{'type':'video','path':f'media/{identity}.mkv','fps':25}],
               'metadata':{'video':identity,'num_frames':2,'width':1920,'height':1080},
               'events':[{'label':'hit','position_ms':40}]}
        cleanup.write_json(source/split/f'{split}.json',{'labels':{},'data':[row]})
        if n == 1:
            cleanup.write_lossless_frames(iter(pixels[:2]),repairs/f'{identity}.mkv',25)
    report = cleanup.prepare(source,reference,tmp_path/'output',repairs,workers=1)
    assert report['actions'] == {'removed_trailing_frame':1,'restored_original_source':1,'retained':1}
    assert report['resolutions'] == [(20,16)]
    assert not report['paper_resolution_satisfied']
    for split in ['train','validation','test']:
        doc=json.loads((tmp_path/'output'/f'{split}.json').read_text())
        row=doc['data'][0]
        assert row['events'] == [{'label':'hit','position_ms':40,'frame':1}]
        assert not (tmp_path/'output'/row['inputs'][0]['path']).is_symlink()
        with av.open(str(tmp_path/'output'/row['inputs'][0]['path'])) as c:
            assert len(list(c.decode(video=0))) == 2


def test_finegym_rebuild_preserves_padding_and_rejects_wrong_source_fps(tmp_path):
    import importlib.util
    import numpy as np
    spec=importlib.util.spec_from_file_location(
        'finegym_preparation',Path(__file__).parents[3]/'tools/convert/prepare_finegym_spotting.py')
    gym=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gym)
    video=tmp_path/'source.mkv'
    gym.write_lossless_frames((np.full((240,320,3),i*20,dtype=np.uint8) for i in range(8)),video,25)
    reference=tmp_path/'reference';reference.mkdir()
    (reference/'class.txt').write_text('start\n')
    row={'video':'source_E_000000_000001','fps':25,'width':640,'height':480,
         'num_frames':4,'events':[{'frame':1,'label':'start'}],
         '_source_info':{'start_frame':2,'end_frame':4,'pad':[1,1],'effective_pad':[1,1]}}
    for split,rows in [('train',[row]),('val',[]),('test',[])]:
        (reference/f'{split}.json').write_text(json.dumps(rows))
    output=tmp_path/'output'
    with pytest.raises(ValueError,match='Missing 1'):
        gym.finalize(reference,output,'test-commit')
    report=gym.prepare_source(video,reference,output,'test-commit',workers=1)
    assert report['clips']==1
    assert gym.finalize(reference,output,'test-commit')['complete']
    record=json.loads((output/'train.json').read_text())['data'][0]
    assert record['metadata']['source_start_frame']==1
    assert record['metadata']['source_end_frame']==5
    assert record['metadata']['num_frames']==4
    assert record['metadata']['source_dimensions']==[320,240]
    assert record['metadata']['reference_source_dimensions']==[640,480]
    assert record['events'][0]['frame']==1
    row['fps']=30
    (reference/'train.json').write_text(json.dumps([row]))
    with pytest.raises(ValueError,match='source FPS'):
        gym.prepare_source(video,reference,output,'test-commit',workers=1)


def test_figureskating_shares_media_and_rebases_exact_frame_intervals(tmp_path):
    import importlib.util
    import numpy as np
    from opensportslib.core.utils.load_annotations import expand_localization_intervals
    spec=importlib.util.spec_from_file_location(
        'skating_preparation',Path(__file__).parents[3]/'tools/convert/prepare_figureskating_spotting.py')
    skating=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(skating)
    # The converter deliberately requires the reference 1080p/25-fps source.
    from lossless_video import write_lossless_frames
    source=tmp_path/'source';source.mkdir()
    write_lossless_frames((np.full((1080,1920,3),i,dtype=np.uint8) for i in range(8)),source/'broadcast.mkv',25)
    reference=tmp_path/'reference'
    for protocol in ['fs_comp','fs_perf']:
        folder=reference/protocol;folder.mkdir(parents=True)
        (folder/'class.txt').write_text('jump\n')
        for index,split in enumerate(['train','val','test']):
            start=index*2
            row={'video':f'broadcast_01_{start:08d}_{start+2:08d}',
                 'num_frames':2,'events':[{'frame':1,'label':'jump'}]}
            (folder/f'{split}.json').write_text(json.dumps([row]))
    output=tmp_path/'output'
    report=skating.prepare(source,reference,output,'test-commit')
    assert report['source_video_count']==1
    assert report['unique_performances']==3
    record=json.loads((output/'fs_comp_test.json').read_text())['data'][0]
    assert record['events'][0]['position_ms']==200
    assert record['events'][0]['source_frame']==5
    segments=expand_localization_intervals(record)
    assert segments[0]['start_time_ms']==160
    assert segments[0]['events'][0]['position_ms']==40
    assert len(list((output/'media').iterdir()))==1


def test_json_to_parquet_and_back_supports_non_video_inputs(tmp_path):
    tracking_path = tmp_path / "test" / "clip_000000.parquet"
    video_path = tmp_path / "test" / "clip_000000.mp4"
    tracking_path.parent.mkdir(parents=True)
    tracking_path.write_bytes(b"tracking-bytes")
    video_path.write_bytes(b"video-bytes")

    payload = {
        "version": "2.0",
        "date": "2026-03-08",
        "task": "action_classification",
        "modalities": ["tracking_parquet", "video"],
        "dataset_name": "mixed_modalities_test",
        "metadata": {"split": "test"},
        "labels": {"action": {"type": "single_label", "labels": ["PASS"]}},
        "data": [
            {
                "id": "sample_000000",
                "inputs": [
                    {"type": "tracking_parquet", "path": "test/clip_000000.parquet"},
                    {"type": "video", "path": "test/clip_000000.mp4", "fps": 25.0},
                ],
                "labels": {"action": {"label": "PASS"}},
                "metadata": {"game_id": "3850"},
            }
        ],
    }
    json_path = tmp_path / "annotations_test.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    parquet_dataset_dir = tmp_path / "converted_dataset"
    forward_result = convert_json_to_parquet(
        json_path=json_path,
        media_root=tmp_path,
        output_dir=parquet_dataset_dir,
        overwrite=True,
    )

    assert forward_result["input_files_added"] == 2

    metadata_df = pd.read_parquet(parquet_dataset_dir / "metadata.parquet")
    row = metadata_df.iloc[0]
    assert "header" in metadata_df.columns
    assert json.loads(row["header"]) == {k: v for k, v in payload.items() if k != "data"}
    assert json.loads(row["sample_payload"]) == payload["data"][0]

    restored_json_path = tmp_path / "restored" / "annotations_test.json"
    restored_media_root = tmp_path / "restored"
    backward_result = convert_parquet_to_json(
        dataset_dir=parquet_dataset_dir,
        output_json_path=restored_json_path,
        extract_media=True,
        output_media_root=restored_media_root,
    )

    restored_payload = json.loads(restored_json_path.read_text(encoding="utf-8"))
    assert restored_payload["modalities"] == payload["modalities"]
    assert restored_payload["data"][0]["inputs"] == payload["data"][0]["inputs"]
    assert backward_result["extracted_input_files"] == 2
    assert (restored_media_root / "test" / "clip_000000.parquet").is_file()
    assert (restored_media_root / "test" / "clip_000000.mp4").is_file()


def test_json_to_parquet_missing_policy_skip_and_raise(tmp_path):
    existing = tmp_path / "clips" / "clip_ok.mp4"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"ok")

    payload = {
        "version": "2.0",
        "task": "action_classification",
        "labels": {},
        "data": [
            {
                "id": "sample_0",
                "inputs": [
                    {"type": "video", "path": "clips/clip_ok.mp4"},
                    {"type": "video", "path": "clips/clip_missing.mp4"},
                ],
            }
        ],
    }
    json_path = tmp_path / "annotations.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    skip_out = tmp_path / "out_skip"
    skip_result = convert_json_to_parquet(
        json_path=json_path,
        media_root=tmp_path,
        output_dir=skip_out,
        missing_policy="skip",
        overwrite=True,
    )
    assert skip_result["input_files_added"] == 1
    assert skip_result["missing_input_files"] == 1

    with pytest.raises(FileNotFoundError):
        convert_json_to_parquet(
            json_path=json_path,
            media_root=tmp_path,
            output_dir=tmp_path / "out_raise",
            missing_policy="raise",
            overwrite=True,
        )


def test_json_to_parquet_absolute_paths_rewrites_sample_payload(tmp_path):
    clip_path = tmp_path / "clips" / "clip_0.mp4"
    clip_path.parent.mkdir(parents=True)
    clip_path.write_bytes(b"clip")

    payload = {
        "version": "2.0",
        "task": "action_classification",
        "labels": {},
        "data": [
            {
                "id": "sample_0",
                "inputs": [{"type": "video", "path": "clips/clip_0.mp4"}],
            }
        ],
    }
    json_path = tmp_path / "annotations.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    out_dir = tmp_path / "out_abs"
    convert_json_to_parquet(
        json_path=json_path,
        media_root=tmp_path,
        output_dir=out_dir,
        keep_relative_paths_in_parquet=False,
        overwrite=True,
    )

    df = pd.read_parquet(out_dir / "metadata.parquet")
    payload_from_parquet = json.loads(df.iloc[0]["sample_payload"])
    assert payload_from_parquet["inputs"][0]["path"] == str(clip_path)


def test_json_to_parquet_size_mode_splits_on_target_tar_size(tmp_path):
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    data = []
    for idx in range(3):
        clip_path = clips_dir / f"clip_{idx}.bin"
        clip_path.write_bytes(b"x" * 3_000)
        data.append({
            "id": f"sample_{idx}",
            "inputs": [{"type": "binary", "path": f"clips/clip_{idx}.bin"}],
        })

    json_path = tmp_path / "annotations.json"
    json_path.write_text(json.dumps({"labels": {}, "data": data}), encoding="utf-8")

    out_dir = tmp_path / "out_size"
    result = convert_json_to_parquet(
        json_path=json_path,
        media_root=tmp_path,
        output_dir=out_dir,
        shard_size=15_000,
        overwrite=True,
    )

    assert result["shard_mode"] == "size"
    assert result["shard_size"] == 15_000
    assert result["num_shards"] == 2
    metadata_df = pd.read_parquet(out_dir / "metadata.parquet")
    assert metadata_df["shard_name"].tolist() == [
        "shard-000000.tar",
        "shard-000000.tar",
        "shard-000001.tar",
    ]


def test_json_to_parquet_oversized_sample_gets_single_shard(tmp_path):
    clip_path = tmp_path / "clips" / "large.bin"
    clip_path.parent.mkdir()
    clip_path.write_bytes(b"x" * 5_000)
    payload = {
        "labels": {},
        "data": [
            {"id": "large", "inputs": [{"type": "binary", "path": "clips/large.bin"}]},
            {"id": "small", "inputs": []},
        ],
    }
    json_path = tmp_path / "annotations.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    out_dir = tmp_path / "out_oversized"
    result = convert_json_to_parquet(
        json_path=json_path,
        media_root=tmp_path,
        output_dir=out_dir,
        shard_size=2_000,
        overwrite=True,
    )

    assert result["num_shards"] == 2
    metadata_df = pd.read_parquet(out_dir / "metadata.parquet")
    assert metadata_df["shard_name"].tolist() == ["shard-000000.tar", "shard-000001.tar"]


def test_json_to_parquet_samples_mode_preserves_count_based_sharding(tmp_path):
    payload = {
        "labels": {},
        "data": [
            {"id": f"sample_{idx}", "inputs": []}
            for idx in range(5)
        ],
    }
    json_path = tmp_path / "annotations.json"
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    out_dir = tmp_path / "out_samples"
    result = convert_json_to_parquet(
        json_path=json_path,
        media_root=tmp_path,
        output_dir=out_dir,
        shard_mode="samples",
        samples_per_shard=2,
        overwrite=True,
    )

    assert result["shard_mode"] == "samples"
    assert result["samples_per_shard"] == 2
    assert result["num_shards"] == 3
    metadata_df = pd.read_parquet(out_dir / "metadata.parquet")
    assert metadata_df["shard_name"].tolist() == [
        "shard-000000.tar",
        "shard-000000.tar",
        "shard-000001.tar",
        "shard-000001.tar",
        "shard-000002.tar",
    ]


def test_parse_shard_size_supports_human_readable_units():
    assert parse_shard_size("500MB") == 500_000_000
    assert parse_shard_size("1GB") == 1_000_000_000
    assert parse_shard_size("1024MiB") == 1024 * 1024**2
    assert parse_shard_size("42") == 42


def test_parquet_to_json_rejects_legacy_schema_without_header(tmp_path):
    dataset_dir = tmp_path / "legacy_dataset"
    shards_dir = dataset_dir / "shards"
    shards_dir.mkdir(parents=True)

    legacy_df = pd.DataFrame(
        [
            {
                "sample_id": "sample_0",
                "sample_index": 0,
                "shard_name": "shard-000000.tar",
                "sample_inputs": json.dumps([{"type": "video", "path": "clips/clip.mp4"}]),
            }
        ]
    )
    legacy_df.to_parquet(dataset_dir / "metadata.parquet", index=False)

    with pytest.raises(ValueError, match="Legacy schemas are no longer supported"):
        convert_parquet_to_json(
            dataset_dir=dataset_dir,
            output_json_path=tmp_path / "reconstructed.json",
        )


@pytest.mark.parametrize('relative', [True, False])
def test_tracab_metadata_companion_roundtrip(tmp_path, relative):
    from tests.helpers.tracab import write_tracab
    descriptor = write_tracab(tmp_path / 'clips' / 'match.dat', sidecar=True)
    descriptor['path'] = 'clips/match.dat'
    descriptor['metadata_path'] = 'clips/match.metadata.json'
    for key in ('tf05_path', 'tf08_path'):
        descriptor[key] = str(Path(descriptor[key]).relative_to(tmp_path))
    payload = {'labels': {}, 'data': [{'id': 'match', 'inputs': [descriptor]}]}
    manifest = tmp_path / 'annotations.json'
    manifest.write_text(json.dumps(payload))
    archive = tmp_path / 'archive'
    convert_json_to_parquet(manifest, tmp_path, archive, keep_relative_paths_in_parquet=relative)
    rows = pd.read_parquet(archive / 'shard_manifest.parquet')
    assert set(rows.file_role) == {'primary', 'metadata', 'tf05', 'tf08'}
    stored = json.loads(pd.read_parquet(archive / 'metadata.parquet').iloc[0].sample_payload)['inputs'][0]
    assert stored['metadata_path'] == ('clips/match.metadata.json' if relative else str(tmp_path / 'clips/match.metadata.json'))
    output = tmp_path / 'restored'
    convert_parquet_to_json(archive, output / 'annotations.json', extract_media=True, output_media_root=output)
    restored = json.loads((output / 'annotations.json').read_text())
    assert restored == payload
    for key in ('path', 'metadata_path', 'tf05_path', 'tf08_path'):
        assert (output / descriptor[key]).read_bytes() == (tmp_path / descriptor[key]).read_bytes()
