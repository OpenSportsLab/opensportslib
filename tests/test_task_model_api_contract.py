import inspect

from opensportslib.apis import ClassificationModel, LocalizationModel


def test_method_signatures_expose_weights_and_no_pretrained_in_signature(
    classification_config_path,
    localization_config_path,
):
    cls_api = ClassificationModel(config=classification_config_path)
    loc_api = LocalizationModel(config=localization_config_path)

    for api in (cls_api, loc_api):
        for method_name in ("load_weights", "train", "infer", "evaluate"):
            sig = inspect.signature(getattr(api, method_name))
            assert "weights" in sig.parameters
            assert "pretrained" not in sig.parameters
            assert any(
                p.kind == inspect.Parameter.VAR_KEYWORD
                for p in sig.parameters.values()
            )

    load_sig = inspect.signature(cls_api.load_weights)
    assert "optimizer" not in load_sig.parameters
    assert "scheduler" not in load_sig.parameters

    infer_sig = inspect.signature(cls_api.infer)
    assert "output_path" not in infer_sig.parameters

    eval_sig = inspect.signature(cls_api.evaluate)
    assert "predictions" not in eval_sig.parameters

    loc_train_sig = inspect.signature(loc_api.train)
    loc_infer_sig = inspect.signature(loc_api.infer)
    loc_eval_sig = inspect.signature(loc_api.evaluate)
    assert "use_ddp" not in loc_train_sig.parameters
    assert "use_ddp" not in loc_infer_sig.parameters
    assert "use_ddp" not in loc_eval_sig.parameters

    save_sig = inspect.signature(cls_api.save_predictions)
    assert "output_path" in save_sig.parameters
    assert "predictions" in save_sig.parameters
    assert save_sig.parameters["predictions"].default is inspect._empty


def test_save_predictions_writes_dict_payload(classification_config_path, tmp_path):
    api = ClassificationModel(config=classification_config_path)
    out_path = tmp_path / "predictions.json"

    saved = api.save_predictions(
        output_path=str(out_path),
        predictions={"items": [{"label": "PASS", "confidence": 0.9}]},
    )

    assert saved == str(out_path)
    assert out_path.exists()


def test_constructor_is_minimal_and_sets_run_id(
    classification_config_path,
    localization_config_path,
):
    cls_sig = inspect.signature(ClassificationModel)
    loc_sig = inspect.signature(LocalizationModel)

    assert list(cls_sig.parameters.keys()) == ["config", "weights"]
    assert list(loc_sig.parameters.keys()) == ["config", "weights"]

    cls_api = ClassificationModel(config=classification_config_path)
    loc_api = LocalizationModel(config=localization_config_path)

    assert cls_api.run_id
    assert loc_api.run_id
