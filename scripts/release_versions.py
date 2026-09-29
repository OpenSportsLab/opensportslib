"""Keep the library version and the server's library pin in sync."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


STABLE_VERSION = re.compile(r"^(?P<version>\d+\.\d+\.\d+)$")
DEV_VERSION = re.compile(r"^(?P<base>\d+\.\d+\.\d+)\.dev(?P<number>\d+)$")
PROJECT_VERSION = re.compile(
    r'(?ms)(^\[project\]\s*.*?^version\s*=\s*")[^"]+("\s*$)'
)
SERVER_PIN = re.compile(r'("opensportslib==)[^"]+("[,\s]*$)', re.MULTILINE)


def next_dev_version(current: str) -> str:
    """Return the next development release for an existing dev version."""
    match = DEV_VERSION.fullmatch(current)
    if not match:
        raise ValueError(f"expected X.Y.Z.devN, got {current!r}")
    return f"{match.group('base')}.dev{int(match.group('number')) + 1}"


def stable_version(tag: str) -> str:
    """Return a stable version from a vX.Y.Z release tag."""
    if not tag.startswith("v"):
        raise ValueError(f"release tag must start with 'v', got {tag!r}")
    version = tag[1:]
    if not STABLE_VERSION.fullmatch(version):
        raise ValueError(f"release tag must have the form vX.Y.Z, got {tag!r}")
    return version


def read_project_version(pyproject: Path) -> str:
    text = pyproject.read_text()
    match = PROJECT_VERSION.search(text)
    if not match:
        raise ValueError(f"project.version not found in {pyproject}")
    return text[match.start(0) + len(match.group(1)) : match.end(0) - len(match.group(2))]


def synchronize(version: str, root_pyproject: Path, server_pyproject: Path) -> None:
    """Write one OpenSportsLib version to both release metadata files."""
    if not (STABLE_VERSION.fullmatch(version) or DEV_VERSION.fullmatch(version)):
        raise ValueError(f"unsupported release version {version!r}")

    root_text = root_pyproject.read_text()
    server_text = server_pyproject.read_text()
    if len(PROJECT_VERSION.findall(root_text)) != 1:
        raise ValueError(f"expected one project.version in {root_pyproject}")
    if len(SERVER_PIN.findall(server_text)) != 1:
        raise ValueError(f"expected one opensportslib dependency pin in {server_pyproject}")

    root_updated = PROJECT_VERSION.sub(rf"\g<1>{version}\g<2>", root_text)
    server_updated = SERVER_PIN.sub(rf"\g<1>{version}\g<2>", server_text)
    root_pyproject.write_text(root_updated)
    server_pyproject.write_text(server_updated)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("bump-dev", "set-stable"))
    parser.add_argument("value", nargs="?", help="vX.Y.Z tag for set-stable")
    parser.add_argument("--root", type=Path, default=Path("pyproject.toml"))
    parser.add_argument("--server", type=Path, default=Path("server/pyproject.toml"))
    args = parser.parse_args()

    if args.command == "bump-dev":
        if args.value is not None:
            parser.error("bump-dev does not accept a value")
        version = next_dev_version(read_project_version(args.root))
    else:
        if args.value is None:
            parser.error("set-stable requires a vX.Y.Z tag")
        version = stable_version(args.value)

    synchronize(version, args.root, args.server)
    print(version)


if __name__ == "__main__":
    main()
