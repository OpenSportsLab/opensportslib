import argparse
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _import_hf_hub():
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'huggingface_hub'. Install it with: pip install huggingface_hub"
        ) from exc
    return HfApi


def _parse_extra_file(value: str) -> tuple[str, str]:
    raw = str(value or "").strip()
    if "=" not in raw:
        raise argparse.ArgumentTypeError(
            "Expected --extra-file in the form LOCAL_PATH=REMOTE_PATH"
        )
    local_path, remote_path = raw.split("=", 1)
    local_path = local_path.strip()
    remote_path = remote_path.strip()
    if not local_path or not remote_path:
        raise argparse.ArgumentTypeError(
            "Expected --extra-file in the form LOCAL_PATH=REMOTE_PATH"
        )
    return local_path, remote_path


def _build_extra_uploads(args) -> list[tuple[str, str]]:
    extra_uploads = list(args.extra_file or [])
    if args.config_path:
        config_target = args.config_path_in_repo or Path(args.config_path).name
        extra_uploads.append((args.config_path, config_target))
    if args.readme_path:
        readme_target = args.readme_path_in_repo or Path(args.readme_path).name
        extra_uploads.append((args.readme_path, readme_target))
    return extra_uploads


def _ensure_repo_and_revision(
    api,
    *,
    repo_id: str,
    repo_type: str,
    revision: str,
    private: bool,
) -> None:
    print(f"[HF] Ensuring {repo_type} repo exists: {repo_id}")
    api.create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=private,
        exist_ok=True,
    )
    if revision == "main":
        return

    commits = api.list_repo_commits(repo_id, repo_type=repo_type)
    base_revision = commits[-1].commit_id if commits else "main"
    print(f"[HF] Ensuring branch exists: {repo_id}@{revision}")
    api.create_branch(
        repo_id=repo_id,
        repo_type=repo_type,
        branch=revision,
        revision=base_revision,
        exist_ok=True,
    )


def _upload_folder(args) -> None:
    HfApi = _import_hf_hub()
    api = HfApi(token=args.token or None)

    folder_path = os.path.abspath(os.path.expanduser(args.folder_path))
    if not os.path.isdir(folder_path):
        raise ValueError(f"Folder does not exist: {folder_path}")

    _ensure_repo_and_revision(
        api,
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        revision=args.revision,
        private=bool(args.private),
    )

    print(f"[HF] Uploading folder: {folder_path}")
    folder_result = api.upload_folder(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        folder_path=folder_path,
        path_in_repo=(args.path_in_repo or None),
        revision=args.revision,
        commit_message=args.commit_message,
        allow_patterns=args.allow_pattern or None,
        ignore_patterns=args.ignore_pattern or None,
        delete_patterns=args.delete_pattern or None,
    )
    print(f"[HF] Folder upload complete: {folder_result}")

    for local_path, remote_path in _build_extra_uploads(args):
        cleaned_local = os.path.abspath(os.path.expanduser(local_path))
        if not os.path.isfile(cleaned_local):
            raise ValueError(f"Extra file does not exist: {cleaned_local}")
        print(f"[HF] Uploading extra file: {cleaned_local} -> {remote_path}")
        file_result = api.upload_file(
            path_or_fileobj=cleaned_local,
            path_in_repo=remote_path,
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            revision=args.revision,
            commit_message=args.commit_message,
        )
        print(f"[HF] Extra file upload complete: {file_result}")


def _upload_file(args) -> None:
    HfApi = _import_hf_hub()
    api = HfApi(token=args.token or None)

    file_path = os.path.abspath(os.path.expanduser(args.file_path))
    if not os.path.isfile(file_path):
        raise ValueError(f"File does not exist: {file_path}")
    if not args.path_in_repo:
        raise ValueError("--path-in-repo is required for file uploads.")

    _ensure_repo_and_revision(
        api,
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        revision=args.revision,
        private=bool(args.private),
    )

    print(f"[HF] Uploading file: {file_path} -> {args.path_in_repo}")
    file_result = api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=args.path_in_repo,
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        revision=args.revision,
        commit_message=args.commit_message,
    )
    print(f"[HF] File upload complete: {file_result}")

    for local_path, remote_path in _build_extra_uploads(args):
        cleaned_local = os.path.abspath(os.path.expanduser(local_path))
        if not os.path.isfile(cleaned_local):
            raise ValueError(f"Extra file does not exist: {cleaned_local}")
        print(f"[HF] Uploading extra file: {cleaned_local} -> {remote_path}")
        extra_result = api.upload_file(
            path_or_fileobj=cleaned_local,
            path_in_repo=remote_path,
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            revision=args.revision,
            commit_message=args.commit_message,
        )
        print(f"[HF] Extra file upload complete: {extra_result}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Upload an OpenSportsLib model artifact to a Hugging Face repo. "
            "Supports both upload_folder and upload_file workflows."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_args(target: argparse.ArgumentParser) -> None:
        target.add_argument("--repo-id", required=True, help="Target Hugging Face repo id, e.g. OpenSportsLab/OSL-VQA-XFOUL-qwen3-8B-VL-lora")
        target.add_argument("--repo-type", default="model", choices=["model", "dataset", "space"], help="Hugging Face repo type (default: model).")
        target.add_argument("--revision", default="main", help="Target branch/revision (default: main).")
        target.add_argument("--private", action="store_true", help="Create the repo as private if it does not exist.")
        target.add_argument("--token", default=None, help="Optional Hugging Face token override.")
        target.add_argument("--commit-message", default="Upload model artifact", help="Commit message to use for uploads.")
        target.add_argument(
            "--extra-file",
            action="append",
            default=[],
            type=_parse_extra_file,
            help="Extra file to upload after the main upload, in the form LOCAL_PATH=REMOTE_PATH. Repeatable.",
        )
        target.add_argument("--config-path", default=None, help="Optional config file to upload as an extra file.")
        target.add_argument("--config-path-in-repo", default="config.yaml", help="Remote path for --config-path (default: config.yaml).")
        target.add_argument("--readme-path", default=None, help="Optional README/model card file to upload as an extra file.")
        target.add_argument("--readme-path-in-repo", default="README.md", help="Remote path for --readme-path (default: README.md).")

    folder_parser = subparsers.add_parser("folder", help="Upload a folder with optional pattern filters.")
    add_common_args(folder_parser)
    folder_parser.add_argument("--folder-path", required=True, help="Local folder to upload.")
    folder_parser.add_argument("--path-in-repo", default="", help="Optional folder prefix inside the remote repo.")
    folder_parser.add_argument("--allow-pattern", action="append", default=[], help="Allow only matching files. Repeatable.")
    folder_parser.add_argument("--ignore-pattern", action="append", default=[], help="Ignore matching files. Repeatable.")
    folder_parser.add_argument("--delete-pattern", action="append", default=[], help="Delete matching remote files before upload. Repeatable.")
    folder_parser.set_defaults(func=_upload_folder)

    file_parser = subparsers.add_parser("file", help="Upload a single file with optional extra files.")
    add_common_args(file_parser)
    file_parser.add_argument("--file-path", required=True, help="Local file to upload.")
    file_parser.add_argument("--path-in-repo", required=True, help="Remote filename/path for the uploaded file.")
    file_parser.set_defaults(func=_upload_file)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
