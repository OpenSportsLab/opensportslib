"""Verify a built SN-GAR spotting release against its contract.

    python tools/convert/verify_sngar_spotting.py --out-root release
    python tools/convert/verify_sngar_spotting.py --out-root release --deep

Exit code 0 iff every check passes. Run it before publishing; the upload script
re-checks sha256 but knows nothing about event counts or modality agreement.

Checks (fast, always):
    1  both repos present with three annotation files each
    2  per-split and per-label event totals match the contract
    3  the two modalities carry identical events, in identical game order
    4  annotations are well formed: sorted, labelled, non-empty, split-tagged
    5  splits are disjoint and cover all 64 games
    6  every referenced payload file exists, and nothing extra ships
    7  no loader caches (*.npy) or __pycache__ in the tree
    8  MANIFEST.sha256 covers every shipped file

Checks (--deep, reads every parquet, several minutes):
    9  each parquet's clock is strictly monotone
   10  every event lands within tolerance of a real tracking row
   11  sha256 of every file matches MANIFEST.sha256

The contract constants below are the published dataset's identity. If a change
to the builder moves them, that is a new dataset version, not a test to be
updated in place.

Usage instructions are at the bottom of this file.
"""

import os
import sys
import json
import hashlib
import argparse
import concurrent.futures

import numpy as np

TRACKING_REPO = "sngar-action-spotting-tracking"
VIDEO_REPO = "sngar-action-spotting-video"
SPLITS = ("train", "valid", "test")

# The published contract: one label per instant, tolerance 10 ms.
# See docs/tools/sngar-spotting.md.
EXPECTED_GAMES = {"train": 45, "valid": 9, "test": 10}
EXPECTED_EVENTS = {"train": 62159, "valid": 12091, "test": 13689}   # 87,939
EXPECTED_LABELS = {
    "PASS": 57516,
    "PLAYER SUCCESSFUL TACKLE": 10943,
    "OUT": 5878,
    "HEADER": 5723,
    "THROW IN": 2598,
    "CROSS": 2175,
    "FREE KICK": 1788,
    "SHOT": 1041,
    "GOAL": 188,
    "HIGH PASS": 89,
}

# Counts for builds that deviate from the published contract, so a deliberate
# variant is still checked rather than skipped. Keyed by
# (deduplicated_events, tolerance_ms).
VARIANT_EVENTS = {
    (True, 34.0): {"train": 62438, "valid": 12176, "test": 13766},   # 88,380
    (False, 10.0): {"train": 62159 + 654, "valid": 12091 + 722, "test": 13689 + 657},
    (False, 34.0): {"train": 65880, "valid": 12865, "test": 14515},  # 93,260
}


class Report:
    def __init__(self):
        self.failures = []

    def check(self, name, ok, detail=""):
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"    {detail}" if detail else ""))
        if not ok:
            self.failures.append(name)
        return ok


# huggingface_hub writes .cache/ into the folder it uploads; not dataset
# content, and never uploaded.
IGNORE_DIRS = {".cache", "__pycache__", ".ipynb_checkpoints"}


def shipped_files(repo_dir, skip=("MANIFEST.sha256",)):
    """Every file that is part of the release, ignoring tooling bookkeeping."""
    out = set()
    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        for f in files:
            if f in skip:
                continue
            out.add(os.path.relpath(os.path.join(root, f), repo_dir))
    return out


def load(repo_dir, split):
    return json.load(open(os.path.join(repo_dir, f"annotations_{split}.json")))


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify(out_root, deep, workers):
    r = Report()
    tracking_dir = os.path.join(out_root, TRACKING_REPO)
    video_dir = os.path.join(out_root, VIDEO_REPO)

    for d in (tracking_dir, video_dir):
        present = all(
            os.path.exists(os.path.join(d, f"annotations_{s}.json")) for s in SPLITS
        )
        if not r.check(f"{os.path.basename(d)}: three annotation files", present):
            return r

    docs = {
        "tracking": {s: load(tracking_dir, s) for s in SPLITS},
        "video": {s: load(video_dir, s) for s in SPLITS},
    }

    meta = docs["tracking"]["train"]["metadata"]
    deduped = bool(meta.get("deduplicated_events"))
    tol = float(meta.get("tolerance_ms", 34.0))
    is_published = (deduped, tol) == (True, 10.0)
    expected_events = (
        EXPECTED_EVENTS if is_published else VARIANT_EVENTS.get((deduped, tol))
    )
    label_rule = "one label per instant" if deduped else "all labels kept"
    if is_published:
        print(f"\nbuild matches the published contract "
              f"({label_rule}, tolerance {tol} ms)\n")
    elif expected_events:
        print(f"\nbuild is a known variant ({label_rule}, tolerance {tol} ms); "
              f"checking against its own counts\n")
    else:
        print(f"\nbuild is an unrecognised variant ({label_rule}, tolerance "
              f"{tol} ms); skipping count checks\n")

    # --- counts -----------------------------------------------------------
    labels_seen = {}
    for split in SPLITS:
        doc = docs["tracking"][split]
        n_events = sum(len(g["events"]) for g in doc["data"])
        r.check(f"{split}: {EXPECTED_GAMES[split]} games",
                len(doc["data"]) == EXPECTED_GAMES[split], f"got {len(doc['data'])}")
        if expected_events:
            r.check(f"{split}: {expected_events[split]} events",
                    n_events == expected_events[split], f"got {n_events}")
        for g in doc["data"]:
            for e in g["events"]:
                labels_seen[e["label"]] = labels_seen.get(e["label"], 0) + 1

    if is_published:
        r.check("per-label totals match the contract", labels_seen == EXPECTED_LABELS,
                str({k: (labels_seen.get(k), v) for k, v in EXPECTED_LABELS.items()
                     if labels_seen.get(k) != v}) if labels_seen != EXPECTED_LABELS else "")

    # --- cross-modality identity -----------------------------------------
    for split in SPLITS:
        t, v = docs["tracking"][split], docs["video"][split]
        same_order = [g["game_id"] for g in t["data"]] == [g["game_id"] for g in v["data"]]
        same_events = all(a["events"] == b["events"] for a, b in zip(t["data"], v["data"]))
        r.check(f"{split}: modalities agree on game order", same_order)
        r.check(f"{split}: modalities carry identical events", same_events)

    # --- well-formedness --------------------------------------------------
    label_set = set(docs["tracking"]["train"]["labels"]["action"]["labels"])
    problems = []
    all_games = set()
    for modality, per_split in docs.items():
        for split, doc in per_split.items():
            for g in doc["data"]:
                if modality == "tracking":
                    all_games.add(g["game_id"])
                if g["split"] != split:
                    problems.append(f"{g['game_id']}: split field is {g['split']}, in {split}")
                if not g["events"]:
                    problems.append(f"{g['game_id']}: no events")
                pos = [e["position_ms"] for e in g["events"]]
                if pos != sorted(pos):
                    problems.append(f"{g['game_id']}: events not sorted by position_ms")
                bad = {e["label"] for e in g["events"]} - label_set
                if bad:
                    problems.append(f"{g['game_id']}: labels outside the declared set: {bad}")
    r.check("annotations well formed", not problems, "; ".join(problems[:3]))

    if deduped:
        collisions = []
        for split in SPLITS:
            for g in docs["tracking"][split]["data"]:
                seen = set()
                for e in g["events"]:
                    if e["position_ms"] in seen:
                        collisions.append(f"{g['game_id']}@{e['position_ms']}")
                    seen.add(e["position_ms"])
        r.check("exactly one label per instant", not collisions, str(collisions[:3]))

    split_sets = [{g["game_id"] for g in docs["tracking"][s]["data"]} for s in SPLITS]
    disjoint = all(
        not (a & b) for i, a in enumerate(split_sets) for b in split_sets[i + 1:]
    )
    r.check("splits are disjoint", disjoint)
    r.check("64 games total", len(all_games) == 64, f"got {len(all_games)}")

    # --- payload presence -------------------------------------------------
    for modality, repo_dir, ext in (
        ("tracking", tracking_dir, ".parquet"), ("video", video_dir, ".mp4")
    ):
        referenced, missing = set(), []
        for split in SPLITS:
            for g in docs[modality][split]["data"]:
                rel = g["inputs"][0]["path"]
                referenced.add(rel)
                if not os.path.isfile(os.path.join(repo_dir, rel)):
                    missing.append(rel)
        on_disk = {f for f in shipped_files(repo_dir, skip=()) if f.endswith(ext)}
        r.check(f"{modality}: every referenced {ext} exists", not missing, str(missing[:3]))
        r.check(f"{modality}: no unreferenced {ext}", not (on_disk - referenced),
                str(sorted(on_disk - referenced)[:3]))

    # --- stray artifacts --------------------------------------------------
    stray = [
        os.path.join(repo_dir, f)
        for repo_dir in (tracking_dir, video_dir)
        for f in shipped_files(repo_dir, skip=())
        if f.endswith((".npy", ".pyc"))
    ]
    r.check("no loader caches or __pycache__ in the tree", not stray, str(stray[:3]))
    r.check("video repo ships no parquets",
            not [f for f in shipped_files(video_dir, skip=()) if f.endswith(".parquet")])

    # --- manifest ---------------------------------------------------------
    manifests = {}
    for repo_dir in (tracking_dir, video_dir):
        name = os.path.basename(repo_dir)
        path = os.path.join(repo_dir, "MANIFEST.sha256")
        if not r.check(f"{name}: MANIFEST.sha256 present", os.path.exists(path)):
            continue
        expected = dict(
            line.rstrip("\n").split("  ", 1)[::-1] for line in open(path)
        )
        on_disk = shipped_files(repo_dir)
        r.check(f"{name}: manifest covers every shipped file",
                set(expected) == on_disk,
                str(sorted(on_disk ^ set(expected))[:3]))
        manifests[repo_dir] = expected

    if not deep:
        return r

    # --- deep: clocks, alignment, hashes ----------------------------------
    import pandas as pd

    print("\ndeep checks (reading every parquet)\n")
    tolerance = float(docs["tracking"]["train"]["metadata"].get("tolerance_ms", 34.0))
    non_monotone, unaligned = [], []
    for split in SPLITS:
        for g in docs["tracking"][split]["data"]:
            path = os.path.join(tracking_dir, g["inputs"][0]["path"])
            times = pd.read_parquet(path, columns=["videoTimeMs"])["videoTimeMs"].to_numpy(
                dtype=np.float64
            )
            if not (np.diff(times) > 0).all():
                non_monotone.append(g["game_id"])
            pos = np.array([e["position_ms"] for e in g["events"]], dtype=np.float64)
            if len(pos):
                idx = np.clip(np.searchsorted(times, pos), 1, len(times) - 1)
                gap = np.minimum(np.abs(times[idx] - pos), np.abs(times[idx - 1] - pos))
                if (gap > tolerance).any():
                    unaligned.append(f"{g['game_id']} (max {gap.max():.1f} ms)")
    r.check("every parquet clock strictly monotone", not non_monotone, str(non_monotone[:3]))
    r.check(f"every event within {tolerance} ms of a tracking row",
            not unaligned, str(unaligned[:3]))

    for repo_dir, expected in manifests.items():
        name = os.path.basename(repo_dir)
        rels = sorted(expected)
        paths = [os.path.join(repo_dir, rel) for rel in rels]
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            digests = list(pool.map(sha256_file, paths))
        bad = [rel for rel, d in zip(rels, digests) if d != expected[rel]]
        r.check(f"{name}: sha256 matches manifest for all {len(rels)} files",
                not bad, str(bad[:3]))

    return r


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out-root", default="release")
    parser.add_argument("--deep", action="store_true",
                        help="also read every parquet and re-hash every file")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    report = verify(args.out_root, args.deep, args.workers)
    print()
    if report.failures:
        print(f"{len(report.failures)} check(s) FAILED:")
        for name in report.failures:
            print(f"  - {name}")
        sys.exit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
#
# Fast checks -- counts, modality agreement, annotation shape, manifest
# coverage. Seconds:
#
#     python verify_sngar_spotting.py --out-root release
#
# Add the deep checks -- read every parquet to confirm the clock is monotone
# and every event lands within tolerance, then re-hash every file against
# MANIFEST.sha256. Several minutes:
#
#     python verify_sngar_spotting.py --out-root release --deep
#
# Exit code is 0 only if every check passes, so this can gate a publish:
#
#     python verify_sngar_spotting.py --out-root release --deep \
#         && python push_sngar_spotting.py --out-root release --yes
