"""Publish the SN-GAR action-spotting datasets to the Hugging Face Hub.

Dry run by default: nothing is created or uploaded without --yes.

Every file is re-hashed against MANIFEST.sha256 before anything is sent, which
catches a payload file rewritten or truncated between build and push where a
size check would not.

Repos are created public with gated="manual". Gating is applied before the repo
is made public, so files are never briefly readable without an approved
request. Pass --private to keep a repo private instead.

Excluded from every upload:
  *.npy               loader feature caches, regenerated on first access
  .cache/             progress state written by the upload client itself
  __pycache__/, .ipynb_checkpoints/, .DS_Store

Usage instructions are at the bottom of this file.
"""

import os
import sys
import fnmatch
import hashlib
import argparse
import concurrent.futures

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm is optional
    def tqdm(iterable=None, total=None, desc=None, **kwargs):
        return iterable if iterable is not None else range(total or 0)

EXCLUDE_GLOBS = ["*.npy", "*.pyc", ".DS_Store"]
# upload_large_folder writes .cache/ into the folder to track its own progress.
# It must never be uploaded.
EXCLUDE_DIRS = {"__pycache__", ".ipynb_checkpoints", ".git", ".cache"}

REPOS = {
    "sngar-action-spotting-tracking": "SNGAR-Action-Spotting-Tracking",
    "sngar-action-spotting-video": "SNGAR-Action-Spotting-Video",
}


def is_excluded(rel_path):
    parts = rel_path.split(os.sep)
    if any(p in EXCLUDE_DIRS for p in parts[:-1]):
        return True
    return any(fnmatch.fnmatch(parts[-1], g) for g in EXCLUDE_GLOBS)


def collect(repo_dir):
    kept, skipped = [], []
    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for name in files:
            path = os.path.join(root, name)
            rel = os.path.relpath(path, repo_dir)
            (skipped if is_excluded(rel) else kept).append((rel, path))
    kept.sort()
    skipped.sort()
    return kept, skipped


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def verify_manifest(repo_dir, kept, num_workers=16):
    """Re-hash every file being uploaded against MANIFEST.sha256.

    Guards the one failure mode that a size check misses: a parquet silently
    rewritten or truncated between build and push.

    Hashing is threaded because this reads the entire release, which is tens
    of gigabytes. sha256 releases the GIL, so threads are sufficient.
    """
    manifest_path = os.path.join(repo_dir, "MANIFEST.sha256")
    if not os.path.exists(manifest_path):
        return ["MANIFEST.sha256 missing - run build_sngar_spotting.py first"]

    expected = {}
    with open(manifest_path) as f:
        for line in f:
            digest, rel = line.rstrip("\n").split("  ", 1)
            expected[rel] = digest

    problems = []
    to_hash = []
    for rel, path in kept:
        if rel == "MANIFEST.sha256":
            continue
        if rel not in expected:
            problems.append(f"not in manifest: {rel}")
            continue
        to_hash.append((rel, path))

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as pool:
        digests = list(tqdm(
            pool.map(sha256_file, [p for _, p in to_hash]),
            total=len(to_hash), desc="verify",
        ))
    for (rel, _), digest in zip(to_hash, digests):
        if digest != expected[rel]:
            problems.append(f"sha256 mismatch: {rel}")
    for rel in expected:
        if rel not in {r for r, _ in kept}:
            problems.append(f"in manifest but missing on disk: {rel}")
    return problems


# Files small enough to fetch and byte-compare when deciding what changed.
# Payload files are compared on size alone; they are large, content-addressed,
# and never rewritten in place by the builder.
SMALL_FILE_BYTES = 64 << 20


def changed_files(api, repo_id, repo_dir, kept):
    """Return the subset of `kept` whose content differs from the Hub.

    Exists because upload_large_folder commits in batches, and a run that
    touches only a few metadata files can still spend dozens of commits
    against the 128/hour repository limit. Re-uploading a handful of changed
    files as one commit costs exactly one.
    """
    try:
        info = api.dataset_info(repo_id, files_metadata=True)
    except Exception:
        return kept, []           # repo absent: everything is new

    remote = {s.rfilename: s.size for s in info.siblings}
    changed, unchanged = [], []
    for rel, path in kept:
        size = os.path.getsize(path)
        if rel not in remote:
            changed.append((rel, path))
            continue
        if remote[rel] is not None and remote[rel] != size:
            changed.append((rel, path))
            continue
        if size > SMALL_FILE_BYTES:
            unchanged.append((rel, path))
            continue
        try:
            local_copy = hf_hub_download(
                repo_id, rel, repo_type="dataset",
                cache_dir=os.path.join(repo_dir, ".cache", "compare"),
            )
            same = open(local_copy, "rb").read() == open(path, "rb").read()
        except Exception:
            same = False
        (unchanged if same else changed).append((rel, path))
    return changed, unchanged


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-root", default="release")
    parser.add_argument("--org", default="OpenSportsLab")
    parser.add_argument("--only", nargs="+", default=None,
                        help="restrict to these local repo dir names")
    parser.add_argument("--repo-suffix", default="",
                        help="appended to each hub repo name, e.g. -v2")
    parser.add_argument("--private", action="store_true",
                        help="keep private instead of public+gated (100 GB cap applies)")
    parser.add_argument("--no-verify", action="store_true",
                        help="skip the sha256 re-check (not recommended)")
    parser.add_argument("--verify-workers", type=int, default=16,
                        help="threads used to re-hash the release before upload")
    parser.add_argument("--changed-only", action="store_true",
                        help="upload only files whose content differs from the\n"
                             "Hub, as a single commit per repo. Use when a build\n"
                             "changed annotations or cards but not the payload.")
    parser.add_argument("--yes", action="store_true",
                        help="actually create repos and upload; without it this is a dry run")
    args = parser.parse_args()

    api = HfApi()
    who = api.whoami()
    print(f"authenticated as {who['name']}\n")

    plans = []
    for local_name, hub_name in REPOS.items():
        if args.only and local_name not in args.only:
            continue
        repo_dir = os.path.join(args.out_root, local_name)
        if not os.path.isdir(repo_dir):
            print(f"skip {local_name}: {repo_dir} does not exist")
            continue

        kept, skipped = collect(repo_dir)
        total = sum(os.path.getsize(p) for _, p in kept)
        repo_id = f"{args.org}/{hub_name}{args.repo_suffix}"

        to_send, unchanged = (
            changed_files(api, repo_id, repo_dir, kept)
            if args.changed_only else (kept, [])
        )
        send_bytes = sum(os.path.getsize(p) for _, p in to_send)

        print(f"{repo_id}")
        print(f"  from      {repo_dir}")
        if args.changed_only:
            print(f"  upload    {len(to_send)} changed files, "
                  f"{send_bytes / 1e9:.2f} GB, in one commit")
            print(f"  unchanged {len(unchanged)} files left as they are")
            for rel, _ in to_send[:10]:
                print(f"              {rel}")
            if len(to_send) > 10:
                print(f"              ... and {len(to_send) - 10} more")
        else:
            print(f"  upload    {len(kept)} files, {total / 1e9:.2f} GB")
        print(f"  exclude   {len(skipped)} files "
              f"({sum(os.path.getsize(p) for _, p in skipped) / 1e9:.2f} GB)")
        print(f"  access    {'private' if args.private else 'public + gated=manual'}")

        if not args.no_verify:
            problems = verify_manifest(repo_dir, kept, args.verify_workers)
            if problems:
                print(f"  MANIFEST  {len(problems)} problem(s):")
                for p in problems[:10]:
                    print(f"              {p}")
                sys.exit(1)
            print(f"  MANIFEST  ok, {len(kept) - 1} files verified")
        print()

        plans.append((repo_id, repo_dir, to_send))

    if not args.yes:
        print("dry run - nothing created or uploaded. re-run with --yes to push.")
        return

    for repo_id, repo_dir, changed in plans:
        if args.changed_only:
            if not changed:
                print(f"{repo_id}: already up to date\n")
                continue
            print(f"{repo_id}: committing {len(changed)} file(s)")
            api.create_commit(
                repo_id=repo_id,
                repo_type="dataset",
                operations=[
                    CommitOperationAdd(path_in_repo=rel, path_or_fileobj=path)
                    for rel, path in changed
                ],
                commit_message="Update annotations and dataset card",
            )
            print(f"done https://huggingface.co/datasets/{repo_id}\n")
            continue

        print(f"creating {repo_id} (private, gated)")
        api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
        api.update_repo_settings(repo_id, repo_type="dataset", gated="manual")
        if not args.private:
            api.update_repo_settings(repo_id, repo_type="dataset", private=False)
            print(f"  -> public + gated=manual")

        print(f"uploading {repo_dir}")
        api.upload_large_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=repo_dir,
            ignore_patterns=["*.npy", "*.pyc", "__pycache__/*", ".ipynb_checkpoints/*",
                             ".cache/*", ".DS_Store"],
        )
        print(f"done https://huggingface.co/datasets/{repo_id}\n")


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------
#
# Sync only what changed, as a single commit per repo. Use after a build that
# rewrote annotations or dataset cards but left the payload alone -- the Hub
# allows 128 repository commits per hour, and a full folder upload spends many
# of them:
#
#     python push_sngar_spotting.py --out-root release --yes --changed-only
#
# Inspect the plan. Prints what would be uploaded, what is excluded and the
# resulting access settings, and verifies the manifest. Creates nothing:
#
#     python push_sngar_spotting.py --out-root release
#
# Publish:
#
#     python push_sngar_spotting.py --out-root release --yes
#
# Publish one modality only, or stage under a suffixed name first:
#
#     python push_sngar_spotting.py --out-root release --yes \
#         --only sngar-action-spotting-tracking
#     python push_sngar_spotting.py --out-root release --yes --repo-suffix -v2
#
# Target a different org, or keep the repos private:
#
#     python push_sngar_spotting.py --out-root release --yes --org <org>
#     python push_sngar_spotting.py --out-root release --yes --private
