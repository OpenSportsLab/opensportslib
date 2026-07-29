"""Run YAML-configured H5 header spotting and save OSL JSON predictions."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from opensportslib.apis import LocalizationModel


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path("/Users/giancos/git/VideoAnnotationTool/test_data")

METHOD_CONFIGS = {
    "distance": REPO_ROOT / "opensportslib/configs/localization/h5_header_distance.yaml",
    "distance_speed": REPO_ROOT / "opensportslib/configs/localization/h5_header_distance_speed.yaml",
    "distance_angle": REPO_ROOT / "opensportslib/configs/localization/h5_header_distance_angle.yaml",
    "distance_speed_angle": REPO_ROOT
    / "opensportslib/configs/localization/h5_header_distance_speed_angle.yaml",
}


def run_method(method: str, config_path: Path, output_dir: Path) -> Path:
    api = LocalizationModel(config=str(config_path))
    predictions = api.infer(use_wandb=False)
    output_path = output_dir / f"h5_header_predictions_{method}.json"
    return Path(api.save_predictions(str(output_path), predictions))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run rule-based H5 header spotting and save OSL JSON predictions. "
            "By default, all four configured methods are run."
        )
    )
    parser.add_argument(
        "--method",
        choices=["all", *METHOD_CONFIGS.keys()],
        default="all",
        help="Header spotting method to run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where prediction JSON files will be saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("RUN_ID", "h5-header-rule")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    methods = METHOD_CONFIGS.keys() if args.method == "all" else [args.method]
    for method in methods:
        saved_path = run_method(method, METHOD_CONFIGS[method], output_dir)
        print(f"{method}: {saved_path}")


if __name__ == "__main__":
    main()
