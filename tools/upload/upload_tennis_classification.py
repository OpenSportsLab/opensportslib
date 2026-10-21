#!/usr/bin/env python3
"""Publish validated tennis stroke clips with OSL's Parquet/WebDataset uploader."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from string import Template
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import av
from huggingface_hub import CommitOperationAdd, HfApi

from opensportslib.tools import (
    create_dataset_branch_on_hf,
    create_dataset_repo_on_hf,
    upload_dataset_as_parquet_to_hf,
)
from opensportslib.tools.osl_json_to_parquet import parse_shard_size
from tools.convert.tennis_to_classification import CLASSES, digest, write_json

REPO_ID = "OpenSportsLab/OSL-cls-tennis-public"
SPLITS = ("train", "valid", "test")
SOURCE_REPO = "OpenSportsLab/OSL-loc-tennis-public"
SOURCE_COMMIT = "f1cc2de012aa317883d3dc371953d268f2a2c86b"
LICENSE_CARD_COMMIT = "036a3e3eb9019493c5092b8db37ba98143e737ce"


def preflight(root):
    """Validate the entire release locally before contacting the Hub."""
    root = Path(root).resolve()
    report = json.loads((root / "reports/build.json").read_text())
    if report.get("status") != "complete" or not report.get("identity"):
        raise ValueError("Finish the classification clip conversion before uploading")
    if report["settings"]["split_policy"] != "match":
        raise ValueError("The public classification release requires match-disjoint splits")
    if (report["settings"]["before_seconds"] != .5 or report["settings"]["after_seconds"] != .5
            or not report["settings"]["trim_neighbors"]):
        raise ValueError("The public release requires midpoint-trimmed half-second context")
    ids, checksums, matches, counts, dimensions, rates, durations = set(), [], {}, {}, Counter(), set(), []
    for split in SPLITS:
        document = json.loads((root / f"{split}.json").read_text())
        samples = document["data"]
        if (document.get("task") != "action_classification"
                or document.get("conversion_identity") != report["identity"]
                or document["labels"]["action"]["labels"] != list(CLASSES)
                or len(samples) != report["split_counts"][split] or not samples):
            raise ValueError(f"{split}: classification manifest differs from the completed build")
        matches[split], counts[split] = set(), Counter()
        for sample in samples:
            sample_id = sample["id"]
            if sample_id in ids:
                raise ValueError(f"Duplicate sample ID: {sample_id}")
            ids.add(sample_id)
            inputs, metadata = sample["inputs"], sample["metadata"]
            conversion = metadata["conversion"]
            if len(inputs) != 1 or inputs[0].get("type") != "video":
                raise ValueError(f"{sample_id}: expected one video input")
            relative = Path(inputs[0]["path"])
            path = (root / relative).resolve()
            if (relative.is_absolute() or not path.is_relative_to(root / "clips" / split)
                    or path.suffix != ".mp4" or not path.is_file()):
                raise ValueError(f"{sample_id}: missing or unsafe clip path")
            receipt = json.loads((root / "receipts" / f"{sample_id}.json").read_text())
            checksum = digest(path)
            if (receipt.get("identity") != report["identity"] or receipt.get("sample") != sample
                    or conversion.get("identity") != report["identity"]
                    or receipt.get("output_sha256") != checksum
                    or receipt.get("source_sha256") != conversion["source_video_sha256"]
                    or conversion["source_manifest_sha256"]
                    != report["source_manifests_sha256"][metadata["source_split"]]):
                raise ValueError(f"{sample_id}: checksum receipt or conversion identity differs")
            label = sample["labels"]["action"]["label"]
            if label not in CLASSES or not 0 <= conversion["event_offset_ms"] < conversion["duration_seconds"] * 1000:
                raise ValueError(f"{sample_id}: invalid label or target outside clip")
            with av.open(str(path)) as container:
                stream = container.streams.video[0]
                if (stream.frames != conversion["frame_count"]
                        or (stream.width, stream.height) != (conversion["width"], conversion["height"])
                        or stream.duration is None
                        or abs(float(stream.duration * stream.time_base) - conversion["duration_seconds"])
                        > max(.001, 2 * float(stream.time_base))):
                    raise ValueError(f"{sample_id}: MP4 header differs from validated clip")
            matches[split].add(metadata["base_video"])
            counts[split][label] += 1
            checksums.append(f"{checksum}  {relative.as_posix()}\n")
            dimensions[(conversion["width"], conversion["height"])] += 1
            rates.add(conversion["source_fps"])
            durations.append(conversion["duration_seconds"])
        if (counts[split] != report["class_counts"][split]
                or sorted(matches[split]) != report["matches"][split]):
            raise ValueError(f"{split}: label or source-match counts differ from build report")
        print(f"Checked {split}: {len(samples):,} clips, {len(matches[split])} matches", flush=True)
    if any(matches[a] & matches[b] for a, b in (("train", "valid"), ("train", "test"), ("valid", "test"))):
        raise ValueError("Source matches overlap between splits")
    excluded = json.loads((root / "reports/excluded_events.json").read_text())
    if (len(ids) != report["expected_clips"] or len(excluded) != report["excluded_events"]
            or len(ids) + len(excluded) != report["requested_events"]):
        raise ValueError("Clip/exclusion totals differ from the completed build")
    return dict(report=report, excluded=excluded, checksums=checksums,
                dimensions=dimensions, rates=rates, durations=durations,
                manifest_sha256={split: digest(root / f"{split}.json") for split in SPLITS})


def render_readme(summary, repo_id, revision):
    report = summary["report"]
    settings = report["settings"]
    rows = "\n".join(f"| `{split}` | {report['split_counts'][split]:,} | {len(report['matches'][split])} |" for split in SPLITS)
    classes = "\n".join(f"| {index} | `{label}` | " + " | ".join(
        f"{report['class_counts'][split].get(label, 0):,}" for split in SPLITS) + " |"
        for index, label in enumerate(CLASSES))
    excluded_rows = "\n".join(f"| `{row['source_record_id']}` | `{row['event']['label']}` | "
        f"{row['event']['position_ms'] / 1000:g} | {row['source_duration_seconds']:g} |" for row in summary["excluded"])
    template = Template(Path(__file__).with_name("tennis_classification_README.md").read_text())
    return template.substitute(
        repo_id=repo_id, revision=revision, source_repo=SOURCE_REPO, source_commit=SOURCE_COMMIT,
        license_card_commit=LICENSE_CARD_COMMIT, identity=report["identity"], split_rows=rows,
        class_rows=classes, total=f"{report['expected_clips']:,}", excluded_count=len(summary["excluded"]),
        excluded_rows=excluded_rows or "| None | — | — | — |",
        before=settings["before_seconds"], after=settings["after_seconds"],
        seed=settings["seed"], fraction=settings["validation_fraction"], crf=settings["crf"],
        dimensions=", ".join(f"{w}×{h}" for w, h in sorted(summary["dimensions"])),
        rates=", ".join(sorted(summary["rates"])),
        shortest=f"{min(summary['durations']):.3f}", longest=f"{max(summary['durations']):.3f}",
        validation_matches="\n".join(f"- `{name}`" for name in report["matches"]["valid"]),
    )


def publish_documents(root, summary, repo_id, revision):
    """Publish the card and machine-readable provenance only after every split."""
    with tempfile.TemporaryDirectory(prefix="osl_tennis_card_") as temporary:
        staging = Path(temporary)
        report = {key: value for key, value in summary["report"].items()
                  if key in {"status", "identity", "settings", "source_manifests_sha256", "matches",
                             "requested_events", "expected_clips", "excluded_events", "class_counts", "split_counts"}}
        report.update(source_repo_id=SOURCE_REPO, source_revision="swing-data", source_commit=SOURCE_COMMIT,
                      classification_manifests_sha256=summary["manifest_sha256"],
                      license_card_commit=LICENSE_CARD_COMMIT)
        write_json(staging / "reports/conversion.json", report)
        write_json(staging / "reports/excluded_events.json", summary["excluded"])
        (staging / "checksums.sha256").write_text("".join(sorted(summary["checksums"])))
        operations = [CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=str(root / "README.md"))]
        operations.extend(CommitOperationAdd(path_in_repo=path.relative_to(staging).as_posix(), path_or_fileobj=str(path))
                          for path in sorted(staging.rglob("*")) if path.is_file())
        return HfApi().create_commit(repo_id=repo_id, repo_type="dataset", revision=revision,
                                    operations=operations, commit_message="Document tennis stroke classification release")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/tennis-classification"))
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--shard-size", default="1GB")
    parser.add_argument("--check-only", action="store_true", help="Validate and write the local README; no Hub calls")
    args = parser.parse_args(argv)
    try:
        root = args.data_root.resolve()
        shard_size = parse_shard_size(args.shard_size)
        summary = preflight(root)
        (root / "README.md").write_text(render_readme(summary, args.repo_id, args.revision))
        print(f"Prepared {root / 'README.md'}", flush=True)
        if args.check_only:
            print("Local checks passed. Nothing uploaded.")
            return 0
        create_dataset_repo_on_hf(args.repo_id, progress_cb=print)
        if args.revision != "main":
            create_dataset_branch_on_hf(args.repo_id, args.revision, progress_cb=print)
        for split in SPLITS:
            result = upload_dataset_as_parquet_to_hf(
                args.repo_id, str(root / f"{split}.json"), revision=args.revision, split=split,
                shard_mode="size", shard_size=shard_size,
                commit_message=f"Upload tennis stroke classification: {split}", progress_cb=print,
            )
            print(json.dumps(result, indent=2, default=str), flush=True)
        commit = publish_documents(root, summary, args.repo_id, args.revision)
        print(f"Uploaded all splits and README: {commit}", flush=True)
        return 0
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
