from pathlib import Path

from opensportslib.apis.classification import ClassificationModel
from tests.helpers.pipeline_stubs import install_classification_pipeline_stub


def _exercise(assets, monkeypatch, tmp_path):
    install_classification_pipeline_stub(monkeypatch, tmp_path)
    monkeypatch.setenv("OSL_PRETRAINED_WEIGHTS", "0")
    monkeypatch.setenv("WANDB_MODE", "disabled")
    api = ClassificationModel(config=assets["config"])
    checkpoint = api.train(use_wandb=False)
    assert checkpoint and Path(checkpoint).exists()
    predictions = api.infer(weights=checkpoint, use_wandb=False)
    assert predictions["task"] == "action_classification"
    assert isinstance(api.evaluate(use_wandb=False), dict)


def test_classification_synthetic_subset_pipeline(classification_integration_assets, monkeypatch, tmp_path):
    assets = classification_integration_assets
    install_classification_pipeline_stub(monkeypatch, tmp_path)
    api = ClassificationModel(config=assets["config"])
    checkpoint = api.train(train_set=assets["train"], valid_set=assets["valid"], use_wandb=False)
    assert checkpoint and Path(checkpoint).exists()
    assert api.infer(test_set=assets["test"], weights=checkpoint, use_wandb=False)["task"] == "action_classification"
    assert isinstance(api.evaluate(test_set=assets["test"], use_wandb=False), dict)


def test_classification_public_format_pipeline(classification_public_dataset_assets, monkeypatch, tmp_path):
    _exercise(classification_public_dataset_assets, monkeypatch, tmp_path)



def test_skeleton_cache_classification_real_graph_model(tmp_path):
    """Cached extraction feeds the actual GAR model and receives gradients."""
    import json
    import numpy as np
    import torch
    from opensportslib.core.config.editable import Config
    from opensportslib.core.config.runtime_adapter import dict_to_namespace
    from opensportslib.datasets.builder import build_dataset
    from opensportslib.models.builder import build_model
    from opensportslib.preprocessing import SkeletonFrame
    from opensportslib.preprocessing.cache import write_cache
    from opensportslib.core.utils.data import tracking_collate_fn

    frames = []
    for i in range(6):
        frame = SkeletonFrame.empty(2,timestamp=i/25,sample_time=i/25,frame_index=i,
                                     pts=i,time_base=(1,25),shot_id=0,image_size=(100,80))
        frame.player_valid[0]=True;frame.track_ids[0]=0
        frame.boxes[0]=[10,10,30,60];frame.joints[0]=20
        frame.joint_valid[0]=True;frame.confidence[0]=1
        frames.append(frame)
    write_cache(tmp_path/'pose',frames,{'test':True},2)
    annotation=tmp_path/'train.json'
    annotation.write_text(json.dumps({'labels':{'action':{'labels':['swing','serve']}},'data':[
        {'id':'stroke','inputs':[{'type':'skeleton_cache','path':'pose'}],
         'labels':{'action':{'label':'swing'}}}]}))
    payload=Config.from_file('examples/tennis/classification_skeleton.yaml').get_config()
    payload['SYSTEM']['device']='cpu'
    payload['DATA']['common']['classes']=['swing','serve']
    payload['DATA']['common']['splits']['train']['source_path']=str(tmp_path)
    payload['DATA']['inputs']['video']['sampling']['num_frames']=4
    payload['MODEL']['components']['tracking_encoder']['params']['num_layers']=1
    payload['MODEL']['components']['task_head']['params']['num_classes']=2
    config=dict_to_namespace(payload)
    dataset=build_dataset(config,str(annotation),split='train')
    batch=tracking_collate_fn([dataset[0]])
    model,_=build_model(config,'cpu')
    logits=model(batch)
    assert logits.shape==(1,2)
    torch.nn.functional.cross_entropy(logits,batch['labels']).backward()
    assert model.backbone.node_encoder.weight.grad is not None
    assert model.backbone.node_encoder.in_features==73
    # Missing nodes are absent from graph computation, including trailing empty graphs.
    model.eval()
    batch['node_mask'][:]=False
    assert torch.isfinite(model(batch)).all()
