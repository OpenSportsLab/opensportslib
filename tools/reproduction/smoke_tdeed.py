"""Repeat a small real-data T-DEED run and compare final weights bit for bit."""

import argparse
import copy
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    import torch
    from opensportslib.apis import LocalizationModel
    from opensportslib.core.config.editable import Config
    from opensportslib.core.trainer.tdeed_trainer import atomic_json

    config = Config.from_file(a.config).get_config()
    output = a.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config["SYSTEM"].update(device=a.device)
    config["SYSTEM"].setdefault("reproducibility", {}).update(seed=17, use_seed=True)
    config["TRAIN"]["epochs"] = 2
    config["TRAIN"]["scheduler"]["warm_up_epochs"] = 1
    config["TRAIN"]["execution"]["start_valid_epoch"] = 0
    config["DATA"]["inputs"]["video"]["sampling"]["epoch_num_frames"] = 200
    for split, spec in config["DATA"]["common"]["splits"].items():
        source = Path(spec["annotation_path"])
        manifest = json.loads(source.read_text())
        manifest["data"] = manifest["data"][:1]
        path = output / f"{split}.json"
        atomic_json(path, manifest)
        spec.update(annotation_path=str(path), source_path=str(source.parent.resolve()))
        spec["dataloader"].update(batch_size=2, num_workers=0)
    states = []
    for index in [1, 2]:
        out = output / f"repeat-{index}"
        if (out / "last.pt").exists():
            raise FileExistsError(out)
        config["SYSTEM"]["paths"]["save_dir"] = str(out)
        api = LocalizationModel(Config(copy.deepcopy(config)))
        api.save_dir = str(out)
        checkpoint = api.train(use_wandb=False)
        predictions = api.infer(weights=checkpoint, use_wandb=False)
        metrics = api.evaluate(predictions=predictions, use_wandb=False)
        api.save_predictions(str(out / "predictions.json"), predictions)
        atomic_json(out / "metrics.json", metrics)
        states.append(
            torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        )
        del api
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    assert states[0]["history"] == states[1]["history"], "Repeated histories differ"
    for name, value in states[0]["model_state_dict"].items():
        assert torch.equal(
            value, states[1]["model_state_dict"][name]
        ), f"Repeated weights differ: {name}"
    report = {
        "status": "passed",
        "device": a.device,
        "repeat_final_weights_bit_exact": True,
        "repeat_history_bit_exact": True,
        "paper_reproduced": False,
        "scope": "one real sample per split, two epochs of 200 training frames; correctness check only",
    }
    atomic_json(output / "verification.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
