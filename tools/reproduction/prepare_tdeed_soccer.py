"""Attach pinned T-DEED frame geometry to existing OSL SoccerNet manifests.

Original annotations are preserved in the source release. This writes new local
manifests and keeps their media paths relative to the requested media root.
Optional media download uses immutable Hub revisions and checks free space.
"""

import argparse
import copy
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def normalize(
    source_root, reference_root, output, media_root, dataset, fallback_root=None
):
    from opensportslib.core.trainer.tdeed_trainer import atomic_json

    output.mkdir(parents=True, exist_ok=True)
    assets = []
    for split, native in [("train", "train"), ("valid", "val"), ("test", "test")]:
        source = source_root / (split + ".json")
        if source.exists():
            document = json.loads(source.read_text())
            rows = document["data"]
        else:
            import pyarrow.parquet as pq

            source = source_root / split / "metadata.parquet"
            rows = [
                json.loads(r["sample_payload"])
                for r in pq.read_table(source).to_pylist()
            ]
            document = {"version": "2.0", "task": "localization"}
        refs = {
            r["video"]: r
            for r in json.loads(
                (
                    reference_root
                    / ("val.json" if split == "valid" else split + ".json")
                ).read_text()
            )
        }
        records = []
        for row in rows:
            row = copy.deepcopy(row)
            inp = row["inputs"][0]
            path = PurePosixPath(inp["path"])
            selected_root = media_root
            selected_revision = (
                "e8859094c746ce2075f6a398e7f8de18939abcad"
                if dataset == "soccernetball"
                else (
                    "8c98e425a8810602a806b4f17d42407354ebbd06"
                    if "720p" in path.name
                    else "24fa021b2e74fe8ee76112a72ed942c61c268038"
                )
            )
            fallback_reason = None
            if dataset == "soccernet" and abs(float(inp["fps"]) - 25) > 0.001:
                if fallback_root is None:
                    raise ValueError(
                        "Non-25-fps 224p source requires --fallback-source-root with the 720p manifests"
                    )
                fallback = json.loads((fallback_root / (split + ".json")).read_text())[
                    "data"
                ]
                candidates = [
                    r
                    for r in fallback
                    if PurePosixPath(r["inputs"][0]["path"]).parent == path.parent
                    and PurePosixPath(r["inputs"][0]["path"]).stem.split("_")[0]
                    == path.stem.split("_")[0]
                ]
                if len(candidates) != 1:
                    raise ValueError("Missing unique 25-fps fallback")
                old = Counter(
                    (e["label"], int(e["position_ms"])) for e in row["events"]
                )
                row = copy.deepcopy(candidates[0])
                if old != Counter(
                    (e["label"], int(e["position_ms"])) for e in row["events"]
                ):
                    raise ValueError("Fallback annotations differ")
                fallback_reason = f'224p source metadata has FPS={inp["fps"]}; use original 25fps 720p source'
                inp = row["inputs"][0]
                path = PurePosixPath(inp["path"])
                selected_root = media_root.parent / "source-720p"
                selected_revision = "8c98e425a8810602a806b4f17d42407354ebbd06"
            identity = (
                str(path.parent / ("half" + path.stem.split("_")[0]))
                if dataset == "soccernet"
                else str(path.parent)
            )
            if identity not in refs:
                raise ValueError(f"Unknown benchmark video: {identity}")
            fps = float(inp["fps"])
            if abs(fps - 25) > 0.001:
                raise ValueError("SoccerNet source must retain 25 fps")
            assets.append(
                {
                    "path": inp["path"],
                    "root": str(selected_root),
                    "revision": selected_revision,
                }
            )
            inp["path"] = os.path.relpath(selected_root / inp["path"], output)
            row["id"] = identity
            row.setdefault("metadata", {}).update(
                video=identity,
                num_frames=refs[identity]["num_frames"],
                fps_raw=25,
                reference_commit="d953b4de8a2ad941c565e1683c5b07255cbe644e",
            )
            row["metadata"]["media_revision"] = selected_revision
            if fallback_reason:
                row["metadata"]["media_fallback_reason"] = fallback_reason
            if dataset == "soccernet":
                row["metadata"]["allow_events_beyond_reference_frames"] = True
                row["metadata"][
                    "outside_reference_frame_policy"
                ] = "Retain annotations as scoring ground truth; reference windows stop at num_frames"
            for event in row["events"]:
                # The reference floors the millisecond->native-frame conversion.
                event["frame"] = int(int(event["position_ms"]) / 1000 * 25)
                if event["frame"] < 0 or (
                    dataset != "soccernet"
                    and event["frame"] >= refs[identity]["num_frames"]
                ):
                    raise ValueError(f"{identity}: event beyond benchmark bounds")
            records.append(row)
        if {r["id"] for r in records} != set(refs):
            raise ValueError(f"{split}: incomplete membership")
        atomic_json(output / (split + ".json"), {**document, "data": records})
    return list({(r["path"], r["revision"]): r for r in assets}.values())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["soccernet", "soccernetball"], required=True)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--reference-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--media-root", type=Path, required=True)
    p.add_argument(
        "--fallback-source-root",
        type=Path,
        help="720p-json manifests for irregular-FPS 224p sources",
    )
    p.add_argument("--download-media", action="store_true")
    p.add_argument(
        "--reserve-gb",
        type=float,
        default=500,
        help="Space reserved for other prepared datasets",
    )
    a = p.parse_args()
    assets = normalize(
        a.source_root,
        a.reference_root,
        a.output,
        a.media_root,
        a.dataset,
        a.fallback_source_root,
    )
    if a.download_media:
        from huggingface_hub import HfApi, snapshot_download

        repo, revision = (
            ("SilvioGiancola/OSL-SoccerNet", "24fa021b2e74fe8ee76112a72ed942c61c268038")
            if a.dataset == "soccernet"
            else ("OpenSportsLab/OSL-SNBAS", "720p-2024")
        )
        if a.dataset == "soccernetball":
            raise ValueError(
                "SNBAS media is in WebDataset shards; use the OSL split downloader, then normalize the extracted JSON manifests"
            )
        inventories = {
            r: {
                s.rfilename: s.size
                for s in HfApi()
                .dataset_info(repo, revision=r, files_metadata=True)
                .siblings
            }
            for r in {x["revision"] for x in assets}
        }
        required = sum(
            inventories[r["revision"]][r["path"]]
            for r in assets
            if not (Path(r["root"]) / r["path"]).is_file()
        )
        a.media_root.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(a.media_root).free < required + a.reserve_gb * 1e9:
            raise RuntimeError(
                f"Need {required/1e9:.1f}GB plus {a.reserve_gb}GB reserved before downloading"
            )
        for root, revision in {(r["root"], r["revision"]) for r in assets}:
            snapshot_download(
                repo_id=repo,
                repo_type="dataset",
                revision=revision,
                allow_patterns=[
                    r["path"]
                    for r in assets
                    if (r["root"], r["revision"]) == (root, revision)
                ],
                local_dir=root,
                max_workers=4,
            )
    print(
        json.dumps(
            {
                "dataset": a.dataset,
                "assets": len(assets),
                "missing_media": sum(
                    not (Path(r["root"]) / r["path"]).is_file() for r in assets
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
