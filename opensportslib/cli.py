from __future__ import annotations

import argparse
from typing import Optional

from opensportslib.setup.setup import setup


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="opensportslib")
    parser.add_argument("command", choices=["setup"])
    parser.add_argument("--pyg", action="store_true")
    parser.add_argument("--dali", action="store_true")
    parser.add_argument("--vqa_xvars", action="store_true")
    parser.add_argument("--vqa_qwen", action="store_true")

    args = parser.parse_args(argv)

    if args.command == "setup":
        setup(
            pyg=args.pyg,
            dali=args.dali,
            vqa_xvars=args.vqa_xvars,
            vqa_qwen=args.vqa_qwen
        )
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
