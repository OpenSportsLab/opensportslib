"""Preflight, run and report the local T-DEED experiment campaign.

No publishing or leaderboard submission is performed. An exit status of zero
from train means a run completed, not that paper results were reproduced.
"""

import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def preflight(config, reference_root, *, verify_hashes=False):
    from opensportslib.datasets.tdeed import load_videos

    common = config["DATA"]["common"]
    dataset = common["dataset_name"]
    results, errors = {}, []
    seen = set()
    refs = Path(reference_root) / dataset
    reference_hashes = json.loads(
        (ROOT / "reproduction/tdeed/reference-files.sha256.json").read_text()
    )
    for name in ["class.txt", "train.json", "val.json", "test.json"]:
        path = refs / name
        if not path.is_file() or sha256(path) != reference_hashes.get(
            dataset + "/" + name
        ):
            errors.append(f"Reference integrity failed: {dataset}/{name}")
    classes = common["classes"]
    class_file = refs / "class.txt"
    if not class_file.is_file() or class_file.read_text().splitlines() != classes:
        errors.append("Missing or different pinned reference class ordering")
    manifest_hashes = {}
    for split in ["train", "valid", "test"]:
        spec = common["splits"][split]
        path = Path(spec["annotation_path"])
        reference = refs / ("val.json" if split == "valid" else split + ".json")
        if not path.is_file() or not reference.is_file():
            errors.append(f"{split}: missing manifest or reference: {path}")
            continue
        manifest_hashes[str(path)] = sha256(path)
        native = {v["video"]: v for v in json.loads(reference.read_text())}
        videos = load_videos(path, spec.get("source_path"))
        ids = {v.identity for v in videos}
        if seen & ids:
            errors.append(f"{split}: performance membership leaks across splits")
        seen |= ids
        if set(native) != ids:
            errors.append(f"{split}: membership differs from benchmark")
        missing, wrong, checked = [], [], set()
        dimensions = {}
        doc = json.loads(path.read_text())
        for row in doc["data"]:
            meta = row.get("metadata", {})
            identity = meta.get("video", row["id"])
            dimensions[identity] = (meta.get("width"), meta.get("height"))
        for video in videos:
            if video.identity in native:
                ref = native[video.identity]
                if ref["num_frames"] != video.frames:
                    wrong.append(f"{video.identity}: frame count")
                if "events" in ref and Counter(
                    (e["label"], e["frame"]) for e in ref["events"]
                ) != Counter((e["label"], e["frame"]) for e in video.events):
                    wrong.append(f"{video.identity}: native events")
            if not video.path.is_file():
                missing.append(str(video.path))
            elif verify_hashes and video.sha256 and str(video.path) not in checked:
                if sha256(video.path) != video.sha256:
                    wrong.append(f"{video.identity}: SHA-256")
                checked.add(str(video.path))
            if dataset == "tennis":
                w, h = dimensions.get(video.identity, (0, 0))
                if (w or 0) < 398 or (h or 0) < 224:
                    wrong.append(
                        f"{video.identity}: source below paper resolution (398x224)"
                    )
        if missing:
            errors.append(
                f"{split}: {len(missing)} missing videos; first: {missing[0]}"
            )
        if wrong:
            errors.append(f"{split}: {len(wrong)} mismatches; first: {wrong[0]}")
        results[split] = {
            "videos": len(videos),
            "expected": len(native),
            "missing": len(missing),
            "mismatches": len(wrong),
            "hashes_checked": len(checked),
            "manifest_sha256": manifest_hashes[str(path)],
        }
    if dataset == "soccernetball":
        errors.append(
            "SNBAS 2024 annotation version differs from paper counts (FREE KICK 21 vs 11, GOAL 13 vs 12)"
        )
    auxiliary = config["TRAIN"].get("sampling", {}).get("auxiliary")
    if auxiliary:
        for split, spec in auxiliary["splits"].items():
            path = Path(spec["annotation_path"])
            if not path.is_file():
                errors.append(f"auxiliary {split}: missing manifest {path}")
            else:
                manifest_hashes[str(path)] = sha256(path)
                missing = [
                    v.path
                    for v in load_videos(path, spec.get("source_path"))
                    if not v.path.is_file()
                ]
                if missing:
                    errors.append(f"auxiliary {split}: {len(missing)} missing videos")
    return {
        "ready": not errors,
        "errors": errors,
        "splits": results,
        "manifest_sha256": manifest_hashes,
        "hashes_verified": verify_hashes,
        "scope": "membership/native events/local presence; optional SHA-256, no full decode",
    }


def configured(path, seed, output):
    from opensportslib.core.config.editable import Config

    cfg = Config.from_file(path)
    document = cfg.get_config()
    document["SYSTEM"].setdefault("reproducibility", {}).update(
        seed=seed, use_seed=True
    )
    document["SYSTEM"]["paths"]["save_dir"] = str(output)
    return Config(document, source=str(path))


def run_one(
    config_path, seed, output, reference_root, *, resume=False, verify_hashes=False
):
    from opensportslib.apis import LocalizationModel
    from opensportslib.core.trainer.tdeed_trainer import atomic_json

    config = configured(config_path, seed, output)
    check = preflight(config.get_config(), reference_root, verify_hashes=verify_hashes)
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "preflight.json", check)
    if not check["ready"]:
        return {
            "status": "blocked_data",
            "output": str(output),
            "errors": check["errors"],
        }
    if (output / "results.json").exists():
        if not resume:
            raise FileExistsError(f"Completed run already exists: {output}")
        result = json.loads((output / "results.json").read_text())
        if result.get("manifest_sha256") != check["manifest_sha256"]:
            raise ValueError("Completed run data changed; use a fresh output directory")
        return {"status": "completed", "output": str(output), "results": result}
    checkpoint = output / "last.pt"
    if checkpoint.exists() and not resume:
        raise FileExistsError(f"Use --resume or a new directory: {output}")
    api = LocalizationModel(config=config)
    # Stable run directory; the shared wrapper normally generates an interactive run ID.
    api.save_dir = str(output)
    original_preflight = output / "data-identity.json"
    if (
        original_preflight.exists()
        and json.loads(original_preflight.read_text()) != check["manifest_sha256"]
    ):
        raise ValueError("Dataset manifests changed since this run started")
    atomic_json(original_preflight, check["manifest_sha256"])
    best = api.train(
        weights=str(checkpoint) if checkpoint.exists() else None, use_wandb=False
    )
    predictions = api.infer(weights=best, use_wandb=False)
    api.save_predictions(str(output / "predictions.json"), predictions)
    metrics = api.evaluate(weights=best, use_wandb=False)
    result = {
        "seed": seed,
        "metrics": metrics,
        "checkpoint_sha256": sha256(best),
        "manifest_sha256": check["manifest_sha256"],
        "paper_reproduced": False,
        "note": "Run complete; aggregate independent seeds and compare targets before any reproduction claim.",
    }
    atomic_json(output / "results.json", result)
    return {"status": "completed", "output": str(output), "results": result}


def evaluate_derived(entry, seed, output, campaign_root):
    from opensportslib.apis import LocalizationModel
    from opensportslib.core.trainer.tdeed_trainer import atomic_json

    source = campaign_root / entry["evaluate_from"] / f"seed-{seed}"
    checkpoint = source / "best.pt"
    if not checkpoint.is_file():
        return {"status": "blocked_dependency", "source": str(source)}
    api = LocalizationModel(config=configured(ROOT / entry["config"], seed, output))
    api.save_dir = str(output)
    metrics = api.evaluate(weights=str(checkpoint), use_wandb=False)
    result = {
        "seed": seed,
        "metrics": metrics,
        "checkpoint_sha256": sha256(checkpoint),
        "source_experiment": entry["evaluate_from"],
        "paper_reproduced": False,
    }
    atomic_json(output / "results.json", result)
    return {"status": "completed", "results": result}


def aggregate(ledger, entries):
    """Report independent-seed means, dispersion and signed published-target gaps."""
    import numpy as np

    target_file = json.loads((ROOT / "reproduction/tdeed/targets.json").read_text())
    targets = target_file["targets"]
    output = []
    for entry in entries:
        runs = [
            r
            for r in ledger["runs"]
            if r["experiment"] == entry["id"] and r["status"] == "completed"
        ]
        if not runs:
            continue
        row = {
            "experiment": entry["id"],
            "seeds": [r["seed"] for r in runs],
            "required_seeds_complete": sorted(r["seed"] for r in runs)
            == sorted(entry["seeds"]),
            "paper_recipe_exact": entry.get("paper_recipe_exact", True),
            "comparison": {},
        }
        size = (
            "big" if "_big" in entry["id"] or "challenge2" in entry["id"] else "small"
        )
        for method in ["none", "nms", "soft_nms"]:
            values = [r["results"]["metrics"][method] for r in runs]
            dataset = entry["dataset"]
            if dataset in ["soccernet", "soccernetball"]:
                names = ["tight", "loose"] if dataset == "soccernet" else ["ball"]
                for index, name in enumerate(names):
                    scores = [v[name]["a_mAP"] * 100 for v in values]
                    mean = float(np.mean(scores))
                    target = (
                        targets[dataset][size][index]
                        if dataset == "soccernet"
                        else targets[dataset][
                            "BAS_only" if "bas_only" in entry["id"] else "BAS_AS"
                        ]
                    )
                    row["comparison"][method + "/" + name] = {
                        "mean_percent": mean,
                        "std_percent": float(np.std(scores)),
                        "target_percent": target,
                        "gap_percentage_points": mean - target,
                        "matches_reported_rounding": round(mean, 2) == target,
                    }
            else:
                for tolerance in ["1", "2"]:
                    scores = [v["tolerances"][tolerance]["mAP"] * 100 for v in values]
                    mean = float(np.mean(scores))
                    target = targets[dataset][size][int(tolerance) - 1]
                    if entry["group"] == "controlled-ablation":
                        target = (
                            target_file["ablation_delta1_percent"].get(entry["id"])
                            if tolerance == "1"
                            else None
                        )
                    row["comparison"][method + "/delta_" + tolerance] = {
                        "mean_percent": mean,
                        "std_percent": float(np.std(scores)),
                        "target_percent": target,
                        "gap_percentage_points": (
                            mean - target if target is not None else None
                        ),
                        "matches_reported_rounding": (
                            round(mean, 2) == target if target is not None else None
                        ),
                    }
                    if dataset == "finegym":
                        means = [
                            v["FG-Start"]["tolerances"][tolerance]["mAP"] * 100
                            for v in values
                        ]
                        fgmean = float(np.mean(means))
                        fgtarget = targets["FG-Start"][size][int(tolerance) - 1]
                        row["comparison"][method + "/FG-Start/delta_" + tolerance] = {
                            "mean_percent": fgmean,
                            "std_percent": float(np.std(means)),
                            "target_percent": fgtarget,
                            "gap_percentage_points": fgmean - fgtarget,
                            "matches_reported_rounding": round(fgmean, 2) == fgtarget,
                        }
        output.append(row)
    return output


def select_stage(entries, stage="all", priorities=()):
    """Keep publication stages explicit; priorities retain seed pairing/order."""
    groups = {
        "main": {"released-main", "paper-main"},
        "ablations": {"bas-ablation", "controlled-ablation"},
    }
    selected = [r for r in entries if stage == "all" or r["group"] in groups[stage]]
    unknown = set(priorities) - {r["id"] for r in selected}
    if unknown:
        raise ValueError(f"Priority experiments outside selected stage: {sorted(unknown)}")
    order = {name: i for i, name in enumerate(priorities)}
    return sorted(selected, key=lambda r: order.get(r["id"], len(order)))


def scheduled_runs(entries, readiness=None):
    """Recheck blocked jobs between runs, keeping the GPU on ready work."""
    pending = [(entry, seed) for entry in entries for seed in entry["seeds"]]
    while pending:
        for index, (entry, seed) in enumerate(pending):
            check = readiness(entry, seed) if readiness else {"ready": True}
            if not check["ready"]:
                yield entry, seed, check
                continue
            pending.pop(index)
            yield entry, seed, None
            break
        else:
            # Do not wait forever when all remaining datasets are blocked.
            return


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["plan", "preflight", "run", "campaign", "report"])
    p.add_argument("--config", type=Path)
    p.add_argument(
        "--matrix", type=Path, default=ROOT / "reproduction/tdeed/matrix.json"
    )
    p.add_argument(
        "--reference-root",
        type=Path,
        default=ROOT / "data/spotting-cleanup/references/T-DEED-d953b4d/data",
    )
    p.add_argument("--output", type=Path, default=ROOT / "outputs/tdeed/campaign")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--verify-hashes", action="store_true")
    p.add_argument(
        "--experiment",
        action="append",
        help="Restrict campaign to named matrix entries",
    )
    p.add_argument("--group", action="append", help="Restrict campaign to named groups")
    p.add_argument("--stage", choices=["all", "main", "ablations"], default="all")
    p.add_argument(
        "--priority", action="append", default=[],
        help="Run these experiments first, in argument order",
    )
    p.add_argument(
        "--skip-blocked", action="store_true",
        help="Retry blocked datasets between runs; stop when only blocked jobs remain",
    )
    a = p.parse_args()
    from opensportslib.core.trainer.tdeed_trainer import atomic_json

    a.output.mkdir(parents=True, exist_ok=True)
    if a.action == "preflight":
        if a.config is None:
            p.error("--config is required")
        result = preflight(
            configured(a.config, a.seed, a.output).get_config(),
            a.reference_root,
            verify_hashes=a.verify_hashes,
        )
        atomic_json(a.output / "preflight.json", result)
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if result["ready"] else 2)
    if a.action == "run":
        if a.config is None:
            p.error("--config is required")
        result = run_one(
            a.config,
            a.seed,
            a.output,
            a.reference_root,
            resume=a.resume,
            verify_hashes=a.verify_hashes,
        )
        atomic_json(a.output / "status.json", result)
        print(json.dumps(result, indent=2))
        raise SystemExit(0 if result["status"] == "completed" else 2)
    matrix = json.loads(a.matrix.read_text())
    entries = matrix["experiments"]
    if a.group:
        entries = [r for r in entries if r["group"] in a.group]
    if a.experiment:
        unknown = set(a.experiment) - {r["id"] for r in entries}
        if unknown:
            p.error(f"Unknown experiments: {sorted(unknown)}")
        entries = [r for r in entries if r["id"] in a.experiment]
    try:
        entries = select_stage(entries, a.stage, a.priority)
    except ValueError as error:
        p.error(str(error))
    if a.action == "plan":
        cache, plans = {}, []
        for entry in entries:
            doc = configured(ROOT / entry["config"], 1, a.output).get_config()
            key = json.dumps(
                {
                    "data": doc["DATA"]["common"],
                    "auxiliary": doc["TRAIN"].get("sampling"),
                },
                sort_keys=True,
            )
            if key not in cache:
                cache[key] = preflight(
                    doc, a.reference_root, verify_hashes=a.verify_hashes
                )
            plans.append(
                {
                    "experiment": entry["id"],
                    "seeds": entry["seeds"],
                    "group": entry["group"],
                    "data": cache[key],
                    "recipe_scope": entry.get("recipe_scope", "released-code"),
                }
            )
        result = {
            "experiments": plans,
            "additional_paper_experiments": matrix["additional_paper_experiments"],
            "ready_experiments": sum(r["data"]["ready"] for r in plans),
            "blocked_experiments": sum(not r["data"]["ready"] for r in plans),
            "paper_reproduced": False,
        }
        atomic_json(a.output / "plan.json", result)
        print(
            json.dumps(
                {
                    k: v
                    for k, v in result.items()
                    if k not in ["experiments", "additional_paper_experiments"]
                }
            )
        )
        return
    ledger = {
        "reference_commit": matrix["reference_commit"],
        "runs": [],
        "additional_paper_experiments": matrix["additional_paper_experiments"],
        "paper_reproduced": False,
    }
    ledger["stage"] = a.stage
    ledger["priorities"] = a.priority
    records = {}

    def readiness(entry, seed):
        if entry.get("evaluate_from"):
            source = a.output / entry["evaluate_from"] / f"seed-{seed}" / "best.pt"
            return {"ready": source.is_file(), "errors": [f"Missing checkpoint: {source}"]}
        try:
            return preflight(
                configured(ROOT / entry["config"], seed, a.output).get_config(),
                a.reference_root,
            )
        except Exception:
            # Let run_one record malformed-data failures through the normal
            # per-run exception handler instead of aborting the whole campaign.
            return {"ready": True}

    jobs = scheduled_runs(
        entries, readiness if a.action == "campaign" and a.skip_blocked else None
    )
    for entry, seed, blocked in jobs:
        out = a.output / entry["id"] / f"seed-{seed}"
        item = {"experiment": entry["id"], "seed": seed}
        if blocked is not None:
            item.update(status="blocked_data", output=str(out), errors=blocked["errors"])
        elif a.action == "campaign":
            ledger["active_run"] = {**item, "output": str(out), "status": "running"}
            atomic_json(a.output / "ledger.json", ledger)
            print(json.dumps(ledger["active_run"]), flush=True)
            try:
                item.update(
                    evaluate_derived(entry, seed, out, a.output)
                    if entry.get("evaluate_from")
                    else run_one(
                        ROOT / entry["config"],
                        seed,
                        out,
                        a.reference_root,
                        resume=a.resume,
                        verify_hashes=a.verify_hashes,
                    )
                )
            except Exception as error:
                item.update(
                    status="failed",
                    error=str(error),
                    error_type=type(error).__name__,
                )
        else:
            result = out / "results.json"
            item.update(status="completed" if result.exists() else "not_completed")
            if result.exists():
                item["results"] = json.loads(result.read_text())
        records[(entry["id"], seed)] = item
        ledger["runs"] = list(records.values())
        ledger.pop("active_run", None)
        ledger["aggregates"] = aggregate(ledger, entries)
        atomic_json(a.output / "ledger.json", ledger)
        print(
            json.dumps({k: v for k, v in item.items() if k != "results"}),
            flush=True,
        )
    # Full paper coverage and measured result comparisons remain explicit.
    ledger["all_selected_runs_completed"] = all(
        r["status"] == "completed" for r in ledger["runs"]
    )
    ledger["aggregates"] = aggregate(ledger, entries)
    atomic_json(a.output / "ledger.json", ledger)


if __name__ == "__main__":
    main()
