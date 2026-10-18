from pathlib import Path

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
    from opensportslib.apis._skeleton import spotting_events, EventFinalizer
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
    for window in iter_windows(frames,4,2):
        _, a=model.predict(sequence_batch(window),use_amp=False)
        assert a.shape==(1,4,5)
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


def test_cached_skeleton_dataset_wrapper_and_masked_training(tmp_path):
    import json
    import math
    import torch
    from opensportslib.core.config.editable import Config
    from opensportslib.core.config.runtime_adapter import dict_to_namespace
    from opensportslib.datasets.builder import build_dataset
    from opensportslib.models.builder import build_model
    from opensportslib.preprocessing import SkeletonFrame
    from opensportslib.preprocessing.cache import write_cache

    frames=[SkeletonFrame.empty(2,timestamp=i/25,sample_time=i/25,frame_index=i,
             pts=i,time_base=(1,25),shot_id=0,image_size=(100,80)) for i in range(3)]
    write_cache(tmp_path/'pose',frames,{'settings':{'sample_fps':25}})
    annotations=tmp_path/'train.json'
    annotations.write_text(json.dumps({'labels':{'action':{'labels':['swing']}},'data':[
        {'id':'match','inputs':[{'type':'skeleton_cache','path':'pose'}],
         'events':[{'label':'swing','position_ms':40}]}]}))
    payload=Config.from_file('examples/tennis/localization_skeleton.yaml').get_config()
    payload['SYSTEM']['device']='cpu';payload['DATA']['common']['classes']=['swing']
    payload['DATA']['inputs']['video']['sampling']['clip_len']=4
    payload['DATA']['common']['splits']['train'].update(annotation_path=str(annotations),source_path=str(tmp_path),overlap_len=2)
    payload['MODEL']['components']['tracking_encoder']['params']['num_layers']=1
    config=dict_to_namespace(payload)
    wrapper=build_dataset(config,split='train')
    dataset=wrapper.building_dataset(wrapper.cfg,gpu=0,default_args=wrapper.default_args)
    loader=wrapper.building_dataloader(dataset,wrapper.cfg.dataloader,gpu=0,dali=False)
    model=build_model(config,'cpu')
    output_bias=model._model.head._pred_fine._fc_out._fc_out.bias
    before=output_bias.detach().clone()
    loss=model.epoch(loader,False,optimizer=torch.optim.SGD(model._model.parameters(),lr=.01),fg_weight=1)
    assert math.isfinite(loss)
    assert not torch.equal(before,output_bias.detach())
