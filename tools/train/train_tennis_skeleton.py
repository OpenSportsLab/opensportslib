"""Train the existing skeleton/graph classifier on converted tennis strokes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def training_config(root, *, decoder="opencv", epochs=20, batch_size=2, cache_dir=None):
    from opensportslib.apis import Config

    root = Path(root).resolve()
    report = json.loads((root / "reports" / "build.json").read_text())
    if report.get("status") != "complete":
        raise ValueError("Finish the classification clip conversion before training")
    classes, matches = None, {}
    for split in ("train", "valid", "test"):
        document = json.loads((root / f"{split}.json").read_text())
        labels = document["labels"]["action"]["labels"]
        classes = labels if classes is None else classes
        if labels != classes or document.get("task") != "action_classification":
            raise ValueError(f"{split}: inconsistent classification vocabulary/task")
        if (document.get("conversion_identity") != report["identity"]
                or len(document["data"]) != report["split_counts"][split]):
            raise ValueError(f"{split}: manifest differs from the completed conversion")
        matches[split] = {sample["metadata"]["base_video"] for sample in document["data"]}
        for sample in document["data"]:
            if sample["labels"]["action"]["label"] not in classes:
                raise ValueError(f"{split}: unknown classification label")
            if any(not (root / item["path"]).is_file() for item in sample["inputs"]):
                raise FileNotFoundError(f"Missing generated video for {sample['id']}")
    if len(classes) != 4:
        raise ValueError("Expected the four requested serve/swing classes")
    if any(matches[a] & matches[b] for a, b in (("train", "test"), ("valid", "test"))):
        raise ValueError("Test matches overlap training/validation")
    if report["settings"]["split_policy"] == "match" and matches["train"] & matches["valid"]:
        raise ValueError("Train/validation matches overlap")
    suffix = "_dali" if decoder == "dali" else ""
    template = Path(__file__).resolve().parents[2] / "examples" / "tennis" / f"classification_skeleton{suffix}.yaml"
    config = Config.from_file(template)
    overrides = {"DATA.common.classes": classes,
                 "MODEL.components.task_head.params.num_classes": len(classes),
                 "TRAIN.sampling.use_weighted_sampler": False}
    if cache_dir is not None:
        overrides["MODEL.components.pose_tracking.params.cache"] = {
            "mode": "read_write", "path": str(Path(cache_dir).resolve())}
    config.update(data={"data_root": str(root)},
                  training={"epochs": epochs, "batch_size": batch_size, "num_workers": 0},
                  overrides=overrides)
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/tennis-classification"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--decoder", choices=("opencv", "dali"), default="opencv")
    parser.add_argument("--cache-dir", type=Path, help="Explicitly enable read/write exact-clip skeleton caching")
    parser.add_argument("--check-only", action="store_true", help="Validate data/configuration without constructing pose models or training")
    args = parser.parse_args()
    config = training_config(args.data_root, decoder=args.decoder, epochs=args.epochs,
                             batch_size=args.batch_size, cache_dir=args.cache_dir)
    if args.check_only:
        print("Dataset and skeleton/graph classification configuration validated. Training was not started.")
        return
    from opensportslib.apis import ClassificationModel

    model = ClassificationModel(config=config)
    checkpoint = model.train(use_wandb=False)
    print(f"Best checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
