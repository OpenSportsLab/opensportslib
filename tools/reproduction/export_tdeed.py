"""Export completed FineDiving runs as minimal, verified Hub release folders.

Each release folder contains config.yaml and model.pth only. Verification
evidence is retained separately. Uploads are an explicit separate step.
"""

import argparse
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

os.environ.setdefault("OMP_NUM_THREADS", "2")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
REPO = "OpenSportsLab/OSL-loc-TDEED"


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def portable_config(config, seed):
    config = copy.deepcopy(config)
    config["SYSTEM"]["device"] = "cpu"
    config["SYSTEM"]["paths"]["save_dir"] = "outputs/tdeed"
    config["SYSTEM"]["reproducibility"].update(seed=seed, use_seed=True)
    config["MODEL"]["components"]["video_encoder"]["params"]["pretrained"] = False
    config["DATA"]["common"]["data_root"] = "data/finediving"
    for split, spec in config["DATA"]["common"]["splits"].items():
        spec["annotation_path"] = "${DATA.common.data_root}/" + split + ".json"
        spec["source_path"] = "${DATA.common.data_root}"
        spec["dataloader"]["num_workers"] = 0
    return config


def source_archive(output):
    names = subprocess.check_output(
        ["git", "ls-files", "-z", "opensportslib", "tools/reproduction",
         "reproduction/tdeed", "pyproject.toml", "README.md", "LICENSE",
         "LICENSE-COMMERCIAL"], cwd=ROOT,
    ).decode().split("\0")
    names = sorted(set(filter(None, names)) | {"tools/reproduction/export_tdeed.py"})
    with tarfile.open(output, "w:gz") as archive:
        for name in names:
            path = ROOT / name
            if path.is_file():
                archive.add(path, arcname="opensportslib-source/" + name)


def export_run(run, entry, output, source):
    evidence = output.parent / "evidence" / output.name
    evidence.mkdir(parents=True, exist_ok=True)
    import cv2
    import numpy as np
    import torch
    import yaml
    from opensportslib.apis import Config, LocalizationModel
    from opensportslib.core.trainer.tdeed_trainer import atomic_json
    from opensportslib.datasets.tdeed import load_videos

    result = json.loads((run / "results.json").read_text())
    record = json.loads((run / "run.json").read_text())
    for path, expected in record["recipe"]["implementation_sha256"].items():
        if digest(ROOT / "opensportslib" / path) != expected:
            raise ValueError(f"Training implementation changed: {path}")
    if digest(run / "best.pt") != result["checkpoint_sha256"]:
        raise ValueError(f"Checkpoint integrity failed: {run}")
    state = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    weights = {k: v.detach().cpu().contiguous() for k, v in state["model_state_dict"].items()}
    torch.save({"model_state_dict": weights}, output / "model.pth")
    trained = Config.from_file(ROOT / entry["config"]).get_config()
    for key in ["MODEL", "DATA", "TRAIN"]:
        trained[key] = record["recipe"][key]
    seed = result["seed"]
    trained["SYSTEM"]["reproducibility"].update(seed=seed, use_seed=True)
    (evidence / "training-config.yaml").write_text(yaml.safe_dump(trained, sort_keys=False))
    portable = portable_config(trained, seed)
    (output / "config.yaml").write_text(yaml.safe_dump(portable, sort_keys=False))
    api = LocalizationModel(config=Config.from_file(output / "config.yaml"))
    api.load_weights(str(output / "model.pth"))
    loaded = api.model.state_dict()
    assert set(loaded) == set(weights)
    assert all(torch.equal(weights[k], loaded[k].cpu()) for k in weights)
    spec = trained["DATA"]["common"]["splits"]["test"]
    video = load_videos(Path(spec["annotation_path"]), spec.get("source_path"))[0]
    capture = cv2.VideoCapture(str(video.path))
    frames = []
    for _ in range(100):
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(cv2.resize(frame, (224, 224)), cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise ValueError("Real-clip verification could not decode frames")
    decoded = len(frames)
    frames.extend([np.zeros((224, 224, 3), dtype=np.uint8)] * (100 - decoded))
    clip = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).unsqueeze(0)
    _, scores = api.model.predict(clip, use_amp=False)
    assert scores.shape == (1, 100, 5) and np.isfinite(scores).all()
    # Re-score the published predictions using the original benchmark manifests.
    evaluation = copy.deepcopy(portable)
    evaluation["DATA"] = trained["DATA"]
    scorer = LocalizationModel(config=Config(evaluation))
    predictions = json.loads((run / "predictions.json").read_text())
    measured = scorer.evaluate(predictions=predictions, use_wandb=False)
    expected = result["metrics"][trained["TRAIN"]["selection"]["postprocessing"]]
    assert measured == expected, "Saved prediction metrics differ from recorded results"
    verification = {
        "status": "passed", "strict_state_keys": True, "all_tensors_bit_exact": True,
        "weights_only_loadable": True, "real_clip_frames_decoded": decoded,
        "real_clip_scores_shape": list(scores.shape), "real_clip_scores_finite": True,
        "saved_test_predictions_rescore_exact": True,
        "scope": "Export reload/inference and saved-prediction rescoring; full test neural inference was not repeated",
    }
    torch.load(output / "model.pth", map_location="cpu", weights_only=True)
    atomic_json(evidence / "verification.json", verification)
    for name in ["results.json", "history.json"]:
        shutil.copy2(run / name, evidence / name)
    with gzip.open(evidence / "predictions.json.gz", "wb") as dest:
        dest.write((run / "predictions.json").read_bytes())
    atomic_json(evidence / "environment.json", record["environment"])
    provenance = {
        "experiment": entry["id"], "seed": seed, "training": record,
        "source_checkpoint_sha256": result["checkpoint_sha256"],
        "export_checkpoint_sha256": digest(output / "model.pth"),
        "dataset": "OpenSportsLab/OSL-loc-FineDiving",
        "dataset_revision": "eb57d8ff662bd5c2fcdeb641870658259b03ea45",
        "manifest_sha256": result["manifest_sha256"],
        "local_source_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
        "source_archive_sha256": digest(source), "paper_reproduced": False,
        "checkpoint_selection": "validation mAP; no test-based selection",
    }
    atomic_json(evidence / "provenance.json", provenance)
    shutil.copy2(source, evidence / source.name)
    for name, path in {
        "LICENSE": ROOT / "LICENSE",
        "LICENSE-TDEED": ROOT / "opensportslib/models/utils/impl/tdeed/LICENSE",
        "LICENSE-E2E-Spot": ROOT / "opensportslib/models/utils/impl/tdeed/LICENSE-E2E-Spot",
    }.items():
        shutil.copy2(path, evidence / name)
    size = "big" if entry["id"].endswith("big") else "small"
    tag = f"v1-finediving-{size}-seed{seed}"
    (evidence / "SHA256SUMS").write_text("".join(f"{digest(p)}  {p.name}\n" for p in sorted(output.iterdir()) if p.is_file() and p.name != "SHA256SUMS"))
    return {"revision": tag, "experiment": entry["id"], "seed": seed,
            "folder": str(output), "evidence_folder": str(evidence), "model_sha256": digest(output / "model.pth"),
            "scores": result["metrics"]["soft_nms"]["tolerances"], "verified": True}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--campaign", type=Path, default=ROOT / "outputs/tdeed/campaign")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.output / "opensportslib-source.tar.gz"
    source_archive(source)
    matrix = json.loads((ROOT / "reproduction/tdeed/matrix.json").read_text())
    entries = {e["id"]: e for e in matrix["experiments"]}
    releases = []
    for result in sorted(args.campaign.glob("FineDiving_*/seed-*/results.json")):
        run = result.parent
        tag = f"v1-finediving-{run.parent.name.split('_')[-1]}-seed{json.loads(result.read_text())['seed']}"
        output = args.output / tag
        if output.exists():
            raise FileExistsError(f"Use a fresh export directory: {output}")
        output.mkdir()
        releases.append(export_run(run, entries[run.parent.name], output, source))
        print(json.dumps({"revision": tag, "status": "verified"}), flush=True)
    from opensportslib.core.trainer.tdeed_trainer import atomic_json
    atomic_json(args.output / "releases.json", {"repo_id": REPO, "releases": releases})


if __name__ == "__main__":
    main()
