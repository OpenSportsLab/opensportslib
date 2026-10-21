from pathlib import Path
import pytest

from opensportslib.apis.localization import LocalizationModel
from tests.helpers.pipeline_stubs import install_localization_pipeline_stub


def _exercise(assets, monkeypatch, tmp_path, explicit_paths):
    install_localization_pipeline_stub(monkeypatch, tmp_path)
    monkeypatch.setenv("OSL_PRETRAINED_WEIGHTS", "0")
    monkeypatch.setenv("WANDB_MODE", "disabled")
    api = LocalizationModel(config=assets["config"])
    kwargs = {"train_set": assets["train"], "valid_set": assets["valid"]} if explicit_paths else {}
    checkpoint = api.train(use_wandb=False, **kwargs)
    assert checkpoint and Path(checkpoint).exists()
    infer_kwargs = {"test_set": assets["test"]} if explicit_paths else {}
    assert api.infer(weights=checkpoint, use_wandb=False, **infer_kwargs)["task"] == "localization"
    assert isinstance(api.evaluate(use_wandb=False, **infer_kwargs), dict)


def test_localization_synthetic_subset_pipeline(localization_integration_assets, monkeypatch, tmp_path):
    _exercise(localization_integration_assets, monkeypatch, tmp_path, True)


def test_localization_public_format_pipeline(localization_public_dataset_assets, monkeypatch, tmp_path):
    _exercise(localization_public_dataset_assets, monkeypatch, tmp_path, False)



def test_skeleton_spotting_stream_cache_model_equivalence(tmp_path):
    import numpy as np
    import torch
    from opensportslib.core.config.editable import Config
    from opensportslib.core.config.runtime_adapter import dict_to_namespace
    from opensportslib.models.builder import build_model
    from opensportslib.preprocessing import SkeletonFrame, SkeletonCache
    from opensportslib.preprocessing.cache import write_cache
    from opensportslib.preprocessing.windows import iter_windows, sequence_batch
    from opensportslib.apis._streaming import spotting_events, EventFinalizer
    from opensportslib.metrics.localization_metric import non_maximum_supression, frame_score_events

    frames=[]
    for i in range(11):
        f=SkeletonFrame.empty(2,timestamp=i/25,sample_time=i/25,frame_index=i,
                              pts=i,time_base=(1,25),shot_id=int(i>=7),image_size=(100,80))
        if i%3:
            f.player_valid[0]=True;f.track_ids[0]=int(i>=7)
            f.boxes[0]=[10,10,30,60];f.joints[0]=20
            f.joint_valid[0]=True;f.confidence[0]=1
        frames.append(f)
    write_cache(tmp_path/'pose',frames,{'test':True},3)
    payload=Config.from_file('examples/tennis/localization_skeleton.yaml').get_config()
    payload['SYSTEM']['device']='cpu'
    payload['DATA']['inputs']['video']['sampling']['clip_len']=4
    payload['DATA']['common']['splits']['test']['overlap_len']=2
    payload['MODEL']['components']['tracking_encoder']['params']['num_layers']=1
    config=dict_to_namespace(payload)
    model=build_model(config,'cpu')
    live=list(spotting_events(model,config,iter(frames)))
    cached=list(spotting_events(model,config,iter(SkeletonCache(tmp_path/'pose'))))
    assert live==cached
    num_outputs=len(payload['DATA']['common']['classes'])+1
    for window in iter_windows(frames,4,2):
        _, a=model.predict(sequence_batch(window),use_amp=False)
        assert a.shape==(1,4,num_outputs)
        assert np.isfinite(a).all()
    # Temporal padding cannot influence bidirectional states.
    short=list(iter_windows(frames[:2],4,2))[0]
    b=sequence_batch(short);model._model.eval()
    padded=model._model(b)
    b.x[4:]=1000
    torch.testing.assert_close(padded,model._model(b))
    # Existing NMS compares all original neighbors and preserves ties.
    scores=np.array([[.1,.8],[.1,.9],[.1,.9],[.1,.2],[.1,.7]])
    classes=['swing'];raw=[];got=[]
    finalizer=EventFinalizer(classes,25,2)
    for f,row in zip(frames,scores):
        raw.extend(frame_score_events(row,classes,int(f.timestamp*1000),int(f.timestamp*25)))
        got.extend(finalizer.push(f,row))
    got.extend(finalizer.finish())
    expected=non_maximum_supression([{'events':raw}],2)[0]['events']
    assert got==expected


def test_video_frontend_spotting_wrapper_and_masked_training(tmp_path,monkeypatch):
    import math
    import torch
    from tests.helpers.video_pose import pose_config,PoseBackend
    from opensportslib.core.config.runtime_adapter import dict_to_namespace
    from opensportslib.datasets.builder import build_dataset
    from opensportslib.models.builder import build_model
    monkeypatch.setattr('opensportslib.preprocessing.backend.RTMLibBackend',PoseBackend)
    config=dict_to_namespace(pose_config('localization',tmp_path,count=3))
    wrapper=build_dataset(config,split='train')
    dataset=wrapper.building_dataset(wrapper.cfg,gpu=0,default_args=wrapper.default_args)
    assert type(dataset).__name__=='ActionSpotDataset'
    loader=wrapper.building_dataloader(dataset,wrapper.cfg.dataloader,gpu=0,dali=False)
    model=build_model(config,'cpu')
    output_bias=model._model.head._pred_fine._fc_out._fc_out.bias
    before=output_bias.detach().clone()
    graph_weight=model._model.backbone.encoder.node_encoder.weight
    graph_before=graph_weight.detach().clone()
    loss=model.epoch(loader,False,optimizer=torch.optim.SGD(model._model.parameters(),lr=.01),fg_weight=1)
    assert math.isfinite(loss)
    assert not torch.equal(before,output_bias.detach())
    assert not torch.equal(graph_before,graph_weight.detach())


@pytest.mark.parametrize("route", ["video", "skeleton"])
def test_native_spotting_trains_selects_and_predicts_with_validation_stopping(tmp_path, monkeypatch, route):
    import json
    import torch
    import yaml
    from tests.helpers.video_pose import pose_config, PoseBackend

    monkeypatch.setenv("OSL_PRETRAINED_WEIGHTS", "0")
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setattr("opensportslib.preprocessing.backend.RTMLibBackend", PoseBackend)
    payload = pose_config("localization", tmp_path, count=7)
    payload["DATA"]["inputs"]["video"]["sampling"]["epoch_num_frames"] = 16
    payload["TRAIN"]["epochs"] = 1
    payload["TRAIN"]["execution"].update(criterion_valid="loss", validation_seed=42)
    payload["TRAIN"]["selection"].update(patience=10, min_delta=1e-4)
    payload["TRAIN"]["scheduler"].update(num_epochs=1, warm_up_epochs=0)
    if route == "video":
        components = payload["MODEL"]["components"]
        components.pop("pose_tracking")
        components.pop("tracking_encoder")
        components["video_encoder"] = {"kind": "encoder", "source": {
            "provider": "opensportslib", "registry": "backbone", "name": "rny002_gsm"}, "params": {}}
        payload["MODEL"]["topology"] = [{"from": "video_encoder", "to": "task_head"}]
        payload["IO"]["inputs"]["video"] = "video_encoder"
        payload["DATA"]["inputs"]["video"]["transform"] = {
            "resize": {"height": 32, "width": 40},
            "normalization": {"mean": [.485, .456, .406], "std": [.229, .224, .225]}}
    for spec in payload["DATA"]["common"]["splits"].values():
        spec["results"] = None
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(payload))
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        api = LocalizationModel(config=str(config_path))
        checkpoint = api.train(use_wandb=False)
        selected = torch.load(checkpoint, weights_only=False, map_location="cpu")
        assert selected["epoch"] == 0 and selected["optimizer_state_dict"]["state"]
        summary = json.loads((Path(checkpoint).parent / "training_summary.json").read_text())
        assert summary["epochs_completed"] == summary["selected_epoch"] == 1
        assert summary["stop_reason"] == "max_epochs"
        api.load_weights(checkpoint)
        existing = set(tmp_path.rglob("*.json"))
        predictions = api.infer(weights=checkpoint, use_wandb=False)
        assert predictions["task"] == "localization" and len(predictions["data"]) == 1
        assert set(tmp_path.rglob("*.json")) == existing
    finally:
        torch.set_num_threads(threads)


def test_pose_head_isolates_cuts_and_both_sides_of_padding(tmp_path):
    import torch
    from tests.helpers.video_pose import pose_config,write_video,inject_backend
    from opensportslib.core.config.runtime_adapter import dict_to_namespace
    from opensportslib.datasets.video_raw import read_raw_clip
    from opensportslib.models.builder import build_model
    from opensportslib.models.preprocessors import graph_sequences
    config=dict_to_namespace(pose_config('localization',tmp_path,count=4))
    write_video(tmp_path/'clip.mp4',4,cut=2)
    model=build_model(config,'cpu')
    inject_backend(model.preprocessor)
    frames,meta=read_raw_clip(tmp_path/'clip.mp4',[-1,0,1,2,3,-1])
    rows=model.preprocessor.observations(frames,meta)
    batch=graph_sequences([rows],'cpu')
    model._model.eval()
    with torch.no_grad():
        before=model._model(batch)
        batch.x[6:10]=1000
        batch.x[:2]=1000
        batch.x[10:]=1000
        after=model._model(batch)
    torch.testing.assert_close(before[:,1:3],after[:,1:3],atol=1e-5,rtol=1e-4)
    assert not before[:,0].any() and not before[:,-1].any()


def test_dali_spotting_route_uses_video_dataset_and_raw_metadata(tmp_path,monkeypatch):
    from tests.helpers.video_pose import pose_config,PoseBackend
    from opensportslib.core.config.runtime_adapter import dict_to_namespace
    from opensportslib.datasets.builder import build_dataset
    from opensportslib.datasets.video_raw import read_raw_clip
    from opensportslib.models.builder import build_model
    payload=pose_config('localization',tmp_path)
    payload['DATA']['common']['runtime']['loader_backend']='dali'
    for split in payload['DATA']['common']['splits'].values():
        split['type']=split['type'].replace('Opencv','Dali')
    config=dict_to_namespace(payload)
    def fake_decode(path,indices,clocks,info,device_id):
        frames,_=read_raw_clip(path,indices)
        return {i:f.permute(1,2,0).numpy() for i,f in zip(indices,frames)}
    monkeypatch.setattr('opensportslib.datasets.video_dali.decode_selected',fake_decode)
    monkeypatch.setattr('opensportslib.preprocessing.backend.RTMLibBackend',PoseBackend)
    wrapper=build_dataset(config,split='train')
    dataset=wrapper.building_dataset(wrapper.cfg,gpu=0,default_args=wrapper.default_args)
    assert type(dataset.dataset).__name__=='ActionSpotDataset'
    loader=wrapper.building_dataloader(dataset,wrapper.cfg.dataloader,gpu=0,dali=True)
    batch=next(iter(loader))
    model=build_model(config,'cpu')
    _,probabilities=model.predict(batch,use_amp=False)
    assert probabilities.shape==(1,4,3)
    assert loader.num_workers==0 and batch['video_meta']


def test_localization_video_inspection_preserves_cuts_timing_and_predictions(tmp_path, monkeypatch):
    import av
    import json
    import numpy as np
    import torch
    import yaml
    from tests.helpers.video_pose import pose_config, PoseBackend, write_video, inspection_tool
    from opensportslib.core.config.runtime_adapter import dict_to_namespace
    from opensportslib.models.builder import build_model
    from opensportslib.apis.localization import LocalizationModel

    monkeypatch.setattr('opensportslib.preprocessing.backend.RTMLibBackend', PoseBackend)
    payload = pose_config('localization', tmp_path, count=7)
    write_video(tmp_path / 'clip.mp4', 7, cut=4)
    model = build_model(dict_to_namespace(payload), 'cpu')
    checkpoint = tmp_path / 'weights.pt'
    torch.save({'model_state_dict': model._model.state_dict()}, checkpoint)
    (tmp_path / 'config.yaml').write_text(yaml.safe_dump(payload))
    output = tmp_path / 'inspection'
    tool = inspection_tool()
    report = tool.run('localization', tmp_path / 'clip.mp4', checkpoint, output, device='cpu')
    assert report['status'] == 'complete' and report['observations'] == 7
    rows = [json.loads(line)['skeleton'] for line in (output / 'poses.jsonl').read_text().splitlines()]
    assert [row['frame_index'] for row in rows] == list(range(7))
    assert [row['shot_id'] for row in rows] == [0] * 4 + [1] * 3
    assert rows[4]['track_ids'][0] != rows[3]['track_ids'][0]
    assert report['render']['source_frames'] == report['render']['sampled_source_frames'] == 7
    with av.open(str(tmp_path / 'clip.mp4')) as movie:
        source = [float(frame.pts * frame.time_base) for frame in movie.decode(video=0)]
    with av.open(str(output / 'annotated.mp4')) as movie:
        actual = [float(frame.pts * frame.time_base) for frame in movie.decode(video=0)]
    np.testing.assert_allclose(actual, np.array(source) - source[0], atol=1e-6)
    predictions = json.loads((output / 'predictions.json').read_text())
    # Export/overlay leaves the established spotting event contract unchanged.
    api = LocalizationModel(config=str(tmp_path / 'config.yaml'), weights=str(checkpoint))
    expected = api.infer(video_path=str(tmp_path / 'clip.mp4'), use_wandb=False)
    assert predictions == expected
