"""Train one OpenSportsLib config from an importable Python entry point.

DataLoader workers using the ``spawn`` multiprocessing context must be able to
reload the parent program from a real file. Do not replace this entry point
with a Python heredoc or ``python -c``.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=("classification", "localization"))
    parser.add_argument("config")
    args = parser.parse_args()

    if args.task == "classification":
        from opensportslib.apis import ClassificationModel

        model = ClassificationModel(config=args.config)
    else:
        from opensportslib.apis import LocalizationModel

        model = LocalizationModel(config=args.config)

    model.train(use_wandb=False)


if __name__ == "__main__":
    main()
