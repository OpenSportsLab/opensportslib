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
SERVER_PIN = re.compile(r'("opensportslib==)([^"]+)(")')


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


def development_version(tag: str) -> str:
    """Return the initial development version for a stable release tag."""
    return f"{stable_version(tag)}.dev0"


def next_patch_development_version(tag: str) -> str:
    """Return the next patch's initial development version for a release tag."""
    major, minor, patch = (int(part) for part in stable_version(tag).split("."))
    return f"{major}.{minor}.{patch + 1}.dev0"


def version_base(version: str) -> str:
    """Return X.Y.Z from either a stable or development package version."""
    dev_match = DEV_VERSION.fullmatch(version)
    if dev_match:
        return dev_match.group("base")
    if STABLE_VERSION.fullmatch(version):
        return version
    raise ValueError(f"unsupported release version {version!r}")


def read_project_version(pyproject: Path) -> str:
    text = pyproject.read_text()
    match = PROJECT_VERSION.search(text)
    if not match:
        raise ValueError(f"project.version not found in {pyproject}")
    return text[match.start(0) + len(match.group(1)) : match.end(0) - len(match.group(2))]


def read_server_pin(pyproject: Path) -> str:
    """Return the OpenSportsLib dependency pin from the server metadata."""
    text = pyproject.read_text()
    matches = SERVER_PIN.findall(text)
    if len(matches) != 1:
        raise ValueError(f"expected one opensportslib dependency pin in {pyproject}")
    return matches[0][1]


def assert_synchronized(root_pyproject: Path, server_pyproject: Path) -> str:
    """Ensure the package version and server dependency pin are identical."""
    version = read_project_version(root_pyproject)
    pin = read_server_pin(server_pyproject)
    if version != pin:
        raise ValueError(
            f"version metadata is not synchronized: {root_pyproject} has {version!r}, "
            f"but {server_pyproject} pins {pin!r}"
        )
    return version


def validate_release(tag: str, root_pyproject: Path, server_pyproject: Path) -> str:
    """Validate that metadata is the prepared prerelease for *tag*."""
    expected = development_version(tag)
    current = assert_synchronized(root_pyproject, server_pyproject)
    if not DEV_VERSION.fullmatch(current) or version_base(current) != stable_version(tag):
        raise ValueError(
            f"release tag {tag!r} requires prepared version {expected[:-1]}N, got {current!r}"
        )
    return current


def prepare_development(tag: str, root_pyproject: Path, server_pyproject: Path) -> str:
    """Start the requested release line at X.Y.Z.dev0."""
    target = development_version(tag)
    current = assert_synchronized(root_pyproject, server_pyproject)
    if tuple(map(int, stable_version(tag).split("."))) <= tuple(map(int, version_base(current).split("."))):
        raise ValueError(f"target release {tag!r} must be newer than current version {current!r}")
    synchronize(target, root_pyproject, server_pyproject)
    return target


def advance_development(tag: str, root_pyproject: Path, server_pyproject: Path) -> str:
    """Advance a released development line to its next patch development base."""
    validate_release(tag, root_pyproject, server_pyproject)
    target = next_patch_development_version(tag)
    synchronize(target, root_pyproject, server_pyproject)
    return target


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
    server_updated = SERVER_PIN.sub(rf"\g<1>{version}\g<3>", server_text)
    root_pyproject.write_text(root_updated)
    server_pyproject.write_text(server_updated)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "bump-dev",
            "set-stable",
            "prepare-dev",
            "validate-release",
            "advance-dev",
            "assert-synchronized",
        ),
    )
    parser.add_argument("value", nargs="?", help="vX.Y.Z tag for set-stable")
    parser.add_argument("--root", type=Path, default=Path("pyproject.toml"))
    parser.add_argument("--server", type=Path, default=Path("server/pyproject.toml"))
    args = parser.parse_args()

    if args.command == "bump-dev":
        if args.value is not None:
            parser.error("bump-dev does not accept a value")
        version = next_dev_version(assert_synchronized(args.root, args.server))
        synchronize(version, args.root, args.server)
    elif args.command == "set-stable":
        if args.value is None:
            parser.error("set-stable requires a vX.Y.Z tag")
        version = stable_version(args.value)
        synchronize(version, args.root, args.server)
    elif args.command == "prepare-dev":
        if args.value is None:
            parser.error("prepare-dev requires a vX.Y.Z tag")
        version = prepare_development(args.value, args.root, args.server)
    elif args.command == "validate-release":
        if args.value is None:
            parser.error("validate-release requires a vX.Y.Z tag")
        version = validate_release(args.value, args.root, args.server)
    elif args.command == "advance-dev":
        if args.value is None:
            parser.error("advance-dev requires a vX.Y.Z tag")
        version = advance_development(args.value, args.root, args.server)
    else:
        if args.value is not None:
            parser.error("assert-synchronized does not accept a value")
        version = assert_synchronized(args.root, args.server)
    print(version)


if __name__ == "__main__":
    main()
