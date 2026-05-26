from opensportslib.apis import VQAModel


def test_vqa_infer_and_evaluate_smoke(vqa_config_path):
    api = VQAModel(config=vqa_config_path)
    predictions = api.infer(use_wandb=False)
    assert predictions.get("task") == "vqa"
    assert isinstance(predictions.get("data"), list)
    assert predictions["data"]
    assert "answer_text" in predictions["data"][0]

    metrics = api.evaluate(predictions=predictions, use_wandb=False)
    assert "exact_match" in metrics
    assert "contains_match" in metrics
    assert "token_f1" in metrics
