import argparse
from opensportslib.setup.setup import setup


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "command",
        choices=["setup"]
    )

    parser.add_argument("--pyg", action="store_true")
    parser.add_argument("--dali", action="store_true")

    args = parser.parse_args()

    if args.command == "setup":
        setup(
            pyg=args.pyg,
            dali=args.dali  # reuse flag for simplicity
        )