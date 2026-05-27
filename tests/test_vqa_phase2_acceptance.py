import json
from pathlib import Path

import yaml

from opensportslib.apis import VQAModel


FIXED_SAMPLE_IDS = ["action_1", "action_2"]
MIN_IMPROVEMENT_DELTA = 0.0


def _filter_prediction_payload(predictions, keep_ids):
    data = [row for row in predictions.get("data", []) if row.get("id") in keep_ids]
    return {"task": "vqa", "data": data}


def _write_subset_annotation(src_path: Path, dst_path: Path, keep_ids):
    payload = json.loads(src_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        subset = [row for row in payload["data"] if isinstance(row, dict) and row.get("id") in keep_ids]
        payload["data"] = subset
        dst_payload = payload
    else:
        subset = [row for row in payload if isinstance(row, dict) and row.get("id") in keep_ids]
        dst_payload = subset
    dst_path.write_text(json.dumps(dst_payload), encoding="utf-8")
    return dst_path


def test_phase2_acceptance_baseline_vs_lora(tmp_path):
    root = Path(__file__).resolve().parents[1]
    cfg_src = root / "opensportslib" / "configs" / "vqa" / "xvars_lora.yaml"
    cfg = yaml.safe_load(cfg_src.read_text(encoding="utf-8"))

    test_src = Path(cfg["DATA"]["common"]["splits"]["test"]["annotation_path"])
    train_src = Path(cfg["DATA"]["common"]["splits"]["train"]["annotation_path"])
    valid_src = Path(cfg["DATA"]["common"]["splits"]["valid"]["annotation_path"])

    subset_test = _write_subset_annotation(test_src, tmp_path / "subset_test.json", FIXED_SAMPLE_IDS)
    subset_train = _write_subset_annotation(train_src, tmp_path / "subset_train.json", FIXED_SAMPLE_IDS)
    subset_valid = _write_subset_annotation(valid_src, tmp_path / "subset_valid.json", FIXED_SAMPLE_IDS)

    cfg["SYSTEM"]["paths"]["save_dir"] = str(tmp_path / "phase2_ckpt")
    cfg["SYSTEM"]["paths"]["work_dir"] = str(tmp_path / "phase2_ckpt")
    cfg["SYSTEM"]["paths"]["log_dir"] = str(tmp_path / "phase2_logs")
    cfg["SYSTEM"]["device"] = "cpu"
    cfg["SYSTEM"]["gpu"]["count"] = 0
    cfg["TRAIN"]["execution"]["hf"]["prefer_cuda"] = False
    cfg["DATA"]["common"]["splits"]["test"]["annotation_path"] = str(subset_test)
    cfg["DATA"]["common"]["splits"]["train"]["annotation_path"] = str(subset_train)
    cfg["DATA"]["common"]["splits"]["valid"]["annotation_path"] = str(subset_valid)

    cfg_path = tmp_path / "phase2_acceptance.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    baseline_cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    baseline_cfg["MODEL"]["metadata"]["backend"] = "baseline"
    baseline_cfg["TRAIN"]["execution"]["backend"] = "baseline"
    baseline_cfg_path = tmp_path / "phase2_baseline.yaml"
    baseline_cfg_path.write_text(yaml.safe_dump(baseline_cfg, sort_keys=False), encoding="utf-8")

    baseline_api = VQAModel(config=str(baseline_cfg_path))
    baseline_preds = baseline_api.infer(use_wandb=False)
    baseline_preds = _filter_prediction_payload(baseline_preds, FIXED_SAMPLE_IDS)
    baseline_metrics = baseline_api.evaluate(predictions=baseline_preds, use_wandb=False)

    lora_api = VQAModel(config=str(cfg_path))
    ckpt = lora_api.train(use_wandb=False)
    loaded_api = VQAModel(config=str(cfg_path), weights=ckpt)
    trained_preds = loaded_api.infer(use_wandb=False)
    trained_preds = _filter_prediction_payload(trained_preds, FIXED_SAMPLE_IDS)
    trained_metrics = loaded_api.evaluate(predictions=trained_preds, use_wandb=False)

    tracked_metrics = ["exact_match", "contains_match", "token_f1"]
    improvements = {
        k: float(trained_metrics.get(k, 0.0)) - float(baseline_metrics.get(k, 0.0))
        for k in tracked_metrics
    }

    report = {
        "date": "2026-05-27",
        "fixed_sample_ids": FIXED_SAMPLE_IDS,
        "checkpoint_path": ckpt,
        "baseline_metrics": baseline_metrics,
        "trained_metrics": trained_metrics,
        "improvements": improvements,
        "min_improvement_delta": MIN_IMPROVEMENT_DELTA,
    }
    report_path = tmp_path / "phase2_acceptance_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    assert report_path.exists()
    assert any(delta > MIN_IMPROVEMENT_DELTA for delta in improvements.values()), report
