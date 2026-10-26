"""Run the main-paper queue after verified predecessor results are complete.

status is read-only; watch persists queue state and starts eligible jobs. A
blocked predecessor never counts as completion. Per-recipe verification must
match the config, training implementation and prepared manifests before training.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
CHECKS = ("model_parity", "dataset_alignment", "metric_parity", "repeatability")
IMPLEMENTATION = (
    "opensportslib/models/base/e2e.py",
    "opensportslib/models/backbones/builder.py",
    "opensportslib/models/heads/builder.py",
    "opensportslib/core/trainer/localization_trainer.py",
    "opensportslib/metrics/localization_metric.py",
    "opensportslib/datasets/localization_dataset.py",
)


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_result(path, seed):
    try:
        result = json.loads(path.read_text())
        return result.get("seed") == seed and bool(result.get("metrics"))
    except (OSError, ValueError):
        return False


def predecessor_status(spec, root=ROOT):
    matrix = json.loads((root / spec["matrix"]).read_text())
    required = [(e["id"], seed) for e in matrix["experiments"]
                if e["group"] in spec["groups"] for seed in e["seeds"]]
    missing = [{"experiment": name, "seed": seed} for name, seed in required
               if not read_result(root / spec["output"] / name / f"seed-{seed}" / "results.json", seed)]
    return {"name": spec["name"], "required": len(required),
            "completed": len(required) - len(missing), "missing": missing,
            "complete": bool(required) and not missing}


def selected_jobs(spec, root=ROOT):
    matrix = json.loads((root / spec["matrix"]).read_text())
    return [(entry, seed) for entry in matrix["experiments"]
            if entry["phase"] in spec["phases"] for seed in entry["seeds"]]


def identity_and_data(entry, root=ROOT):
    from opensportslib.apis import Config

    config_path = root / entry["canonical_config"]
    config = Config.from_file(config_path).get_config()
    manifests, errors, seen = {}, [], set()
    for split in ["train", "valid", "test"]:
        spec = config["DATA"]["common"]["splits"][split]
        manifest = root / spec["annotation_path"]
        if not manifest.is_file():
            errors.append(f"{split}: missing manifest {manifest}")
            continue
        manifests[spec["annotation_path"]] = digest(manifest)
        rows = json.loads(manifest.read_text())["data"]
        ids = [r.get("metadata", {}).get("video", r["id"]) for r in rows]
        if not ids or len(set(ids)) != len(ids) or seen & set(ids):
            errors.append(f"{split}: empty, duplicate or leaking membership")
        seen |= set(ids)
        media_root = root / spec["source_path"]
        missing = sum(not (media_root / row["inputs"][0]["path"]).is_file() for row in rows)
        if missing:
            errors.append(f"{split}: {missing} missing media files")
        if entry["dataset"] == "tennis" and any(
            r.get("metadata", {}).get("height", 0) < 224
            or r.get("metadata", {}).get("width", 0) < 398 for r in rows
        ):
            errors.append(f"{split}: media below required resolution")
    identity = {"config_sha256": digest(config_path),
                "implementation_sha256": {name: digest(root / name) for name in IMPLEMENTATION},
                "manifest_sha256": manifests}
    return identity, errors


def readiness(entry, spec, root=ROOT):
    if not entry.get("canonical_config"):
        return {"ready": False, "errors": ["Missing canonical config"]}
    try:
        identity, errors = identity_and_data(entry, root)
    except (OSError, ValueError, KeyError) as error:
        return {"ready": False, "errors": [str(error)]}
    verification_path = root / spec["verification"]
    verification = json.loads(verification_path.read_text()) if verification_path.is_file() else {}
    record = verification.get("experiments", {}).get(entry["id"], {})
    if record.get("status") != "passed" or any(
        record.get(key) != value for key, value in identity.items()
    ) or not all(record.get("checks", {}).get(check) is True for check in CHECKS):
        errors.append("Missing or outdated model/data/metric/repeatability verification")
    return {"ready": not errors, "errors": errors, "identity": identity}


def queue_status(schedule, root=ROOT):
    predecessor = predecessor_status(schedule["predecessor"], root)
    spec = schedule["campaign"]
    jobs = []
    checks = {}
    for entry, seed in selected_jobs(spec, root):
        output = root / spec["output"] / entry["id"] / f"seed-{seed}"
        item = {"experiment": entry["id"], "seed": seed, "output": str(output),
                "config": entry["canonical_config"]}
        if read_result(output / "results.json", seed):
            item["status"] = "completed"
        elif not predecessor["complete"]:
            item["status"] = "waiting_for_tdeed_main"
        elif (output / "failure.json").exists():
            item["status"] = "failed"
        else:
            if entry["id"] not in checks:
                checks[entry["id"]] = readiness(entry, spec, root)
            check = checks[entry["id"]]
            item.update(status="ready" if check["ready"] else "blocked_prerequisites",
                        errors=check["errors"])
        jobs.append(item)
    return {"updated_utc": datetime.now(timezone.utc).isoformat(),
            "predecessor": predecessor, "campaign": spec["name"], "jobs": jobs,
            "completed": sum(j["status"] == "completed" for j in jobs),
            "total": len(jobs)}


def run_job(schedule, experiment, seed):
    from opensportslib.apis import Config, LocalizationModel

    spec = schedule["campaign"]
    matches = [e for e, s in selected_jobs(spec) if e["id"] == experiment and s == seed]
    if len(matches) != 1:
        raise ValueError("Job is not uniquely present in the scheduled main campaign")
    if not predecessor_status(schedule["predecessor"])["complete"]:
        raise RuntimeError("T-DEED main runs are not complete")
    entry = matches[0]
    check = readiness(entry, spec)
    if not check["ready"]:
        raise RuntimeError("; ".join(check["errors"]))
    output = ROOT / spec["output"] / experiment / f"seed-{seed}"
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / "identity.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != check["identity"]:
        raise ValueError("Run config, implementation or manifest identity changed")
    write_json(identity_path, check["identity"])
    doc = Config.from_file(ROOT / entry["canonical_config"]).get_config()
    doc["SYSTEM"]["reproducibility"].update(use_seed=True, seed=seed)
    doc["SYSTEM"]["paths"].update(work_dir=str(output), save_dir=str(output))
    doc["DATA"]["common"]["splits"]["test"]["results"] = str(output / "predictions.json")
    api = LocalizationModel(config=Config(doc))
    api.save_dir = str(output)
    latest = output / "latest_checkpoint.pt"
    best = api.train(weights=str(latest) if latest.is_file() else None, use_wandb=False)
    predictions = api.infer(weights=best, use_wandb=False)
    api.save_predictions(str(output / "predictions.json"), predictions)
    metrics = api.evaluate(predictions=predictions, use_wandb=False)
    if not metrics:
        raise RuntimeError("Evaluation returned no metrics; run cannot be marked completed")
    write_json(output / "results.json", {"seed": seed, "metrics": metrics,
        "checkpoint_sha256": digest(best), "paper_reproduced": False,
        "identity": check["identity"], "scope": entry["recipe_scope"]})


def main():
    import yaml

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["status", "preflight", "watch", "run-job"])
    p.add_argument("--schedule", type=Path, default=ROOT / "reproduction/main-queue.yaml")
    p.add_argument("--output", type=Path, default=ROOT / "outputs/reproduction/main-queue")
    p.add_argument("--experiment")
    p.add_argument("--seed", type=int)
    args = p.parse_args()
    schedule = yaml.safe_load(args.schedule.read_text())
    if args.action == "status":
        print(json.dumps(queue_status(schedule), indent=2))
        return
    if args.action == "preflight":
        entries = {e["id"]: e for e, seed in selected_jobs(schedule["campaign"])}
        print(json.dumps({name: readiness(entry, schedule["campaign"])
                          for name, entry in entries.items()}, indent=2))
        return
    if args.action == "run-job":
        if args.experiment is None or args.seed is None:
            p.error("run-job requires --experiment and --seed")
        run_job(schedule, args.experiment, args.seed)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "worker.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            status = queue_status(schedule)
            status["worker_pid"] = os.getpid()
            ready = next((j for j in status["jobs"] if j["status"] == "ready"), None)
            if ready:
                status["active_job"] = ready
            write_json(args.output / "status.json", status)
            if status["completed"] == status["total"]:
                return
            if ready:
                output = Path(ready["output"])
                output.mkdir(parents=True, exist_ok=True)
                with (output / "run.log").open("a") as log:
                    result = subprocess.run([
                        sys.executable, str(Path(__file__).resolve()), "run-job",
                        "--schedule", str(args.schedule.resolve()),
                        "--experiment", ready["experiment"], "--seed", str(ready["seed"]),
                    ], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                if result.returncode:
                    write_json(output / "failure.json", {"returncode": result.returncode})
            else:
                time.sleep(min(60, max(1, schedule.get("poll_seconds", 60))))


if __name__ == "__main__":
    main()
