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



def test_video_frontend_classification_real_graph_model(tmp_path,monkeypatch):
    import torch
    from tests.helpers.video_pose import pose_config, PoseBackend
    from opensportslib.core.config.runtime_adapter import dict_to_namespace
    from opensportslib.datasets.builder import build_dataset
    from opensportslib.datasets.video_raw import build_video_loader
    from opensportslib.models.builder import build_model
    monkeypatch.setattr('opensportslib.preprocessing.backend.RTMLibBackend',PoseBackend)
    payload=pose_config('classification',tmp_path)
    payload['MODEL']['components']['pose_tracking']['params']['cache']={'mode':'read_write','path':str(tmp_path/'cache')}
    config=dict_to_namespace(payload)
    dataset=build_dataset(config,str(tmp_path/'labels.json'),split='train')
    assert type(dataset).__name__=='VideoDataset'
    batch=next(iter(build_video_loader(dataset,batch_size=1,num_workers=0)))
    model,_=build_model(config,'cpu')
    model.eval()
    logits=model(batch)
    assert logits.shape==(1,2)
    torch.nn.functional.cross_entropy(logits,batch['labels']).backward()
    assert model.backbone.node_encoder.weight.grad is not None
    assert model.head.classifier[1].weight.grad is not None
    assert model.neck.temporal_position_encoding.grad is not None
    assert model.backbone.node_encoder.in_features==73
    assert not list(model.preprocessor.parameters())
    first=logits.detach()
    torch.testing.assert_close(first,model(batch),atol=1e-5,rtol=1e-4)
    assert model.preprocessor.backend.calls==4
    model.preprocessor.backend.absent=True
    model.preprocessor.settings['cache']['mode']='off'
    assert torch.isfinite(model(batch)).all()


def test_multiview_pose_averages_embeddings_before_classification(tmp_path):
    import torch
    from tests.helpers.video_pose import pose_config, write_video, inject_backend
    from opensportslib.core.config.runtime_adapter import dict_to_namespace
    from opensportslib.datasets.video_raw import read_raw_clip, video_collate
    from opensportslib.models.builder import build_model
    config=dict_to_namespace(pose_config('classification',tmp_path))
    model,_=build_model(config,'cpu')
    inject_backend(model.preprocessor)
    model.eval()
    a,ma=read_raw_clip(tmp_path/'clip.mp4',[0,1,2,3])
    b,mb=read_raw_clip(write_video(tmp_path/'other.mp4',width=140,height=100),[0,1,2,3])
    singles=video_collate([{'pixel_values':[a],'video_meta':[ma],'id':'a'},
                           {'pixel_values':[b],'video_meta':[mb],'id':'b'}])
    multi=video_collate([{'pixel_values':[a,b],'video_meta':[ma,mb],'id':'both'}])
    embeddings=[]
    handle=model.head.register_forward_pre_hook(lambda module,args:embeddings.append(args[0].detach().clone()))
    try:
        model(singles)
        model(multi)
    finally:handle.remove()
    torch.testing.assert_close(embeddings[1],embeddings[0].mean(0,keepdim=True))
