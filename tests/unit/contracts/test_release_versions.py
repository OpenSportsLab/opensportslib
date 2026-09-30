import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[3] / "scripts" / "release_versions.py"
SPEC = importlib.util.spec_from_file_location("release_versions", SCRIPT)
release_versions = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(release_versions)


def test_next_dev_version_increments_counter():
    assert release_versions.next_dev_version("0.3.1.dev5") == "0.3.1.dev6"


@pytest.mark.parametrize("version", ["0.3.1", "0.3.1rc1", "invalid"])
def test_next_dev_version_rejects_non_dev_versions(version):
    with pytest.raises(ValueError, match="expected X.Y.Z.devN"):
        release_versions.next_dev_version(version)


def test_stable_version_requires_stable_v_tag():
    assert release_versions.stable_version("v0.3.1") == "0.3.1"
    with pytest.raises(ValueError, match="vX.Y.Z"):
        release_versions.stable_version("v0.3.1.dev1")


def test_development_version_helpers():
    assert release_versions.development_version("v0.4.0") == "0.4.0.dev0"
    assert release_versions.next_patch_development_version("v0.3.1") == "0.3.2.dev0"
    assert release_versions.version_base("0.3.1.dev5") == "0.3.1"
    assert release_versions.version_base("0.3.1") == "0.3.1"
    with pytest.raises(ValueError, match="unsupported release version"):
        release_versions.version_base("invalid")


def test_synchronize_updates_root_and_server_pin_only(tmp_path):
    root = tmp_path / "pyproject.toml"
    server = tmp_path / "server.toml"
    root.write_text('[project]\nname = "opensportslib"\nversion = "0.3.1.dev5"\n')
    server.write_text(
        '[project]\nname = "opensportslib-server"\nversion = "0.1.0"\n'
        'dependencies = [\n  "opensportslib==0.3.1.dev5",\n  "fastapi",\n]\n'
    )

    release_versions.synchronize("0.3.1.dev6", root, server)

    assert 'version = "0.3.1.dev6"' in root.read_text()
    assert 'version = "0.1.0"' in server.read_text()
    assert '"opensportslib==0.3.1.dev6"' in server.read_text()


def test_synchronize_fails_before_writing_when_server_pin_is_missing(tmp_path):
    root = tmp_path / "pyproject.toml"
    server = tmp_path / "server.toml"
    original = '[project]\nversion = "0.3.1.dev5"\n'
    root.write_text(original)
    server.write_text('[project]\nversion = "0.1.0"\n')

    with pytest.raises(ValueError, match="dependency pin"):
        release_versions.synchronize("0.3.1.dev6", root, server)

    assert root.read_text() == original


def test_assert_synchronized_rejects_mismatched_metadata(tmp_path):
    root = tmp_path / "pyproject.toml"
    server = tmp_path / "server.toml"
    root.write_text('[project]\nversion = "0.3.1.dev5"\n')
    server.write_text('dependencies = ["opensportslib==0.3.1.dev4"]\n')

    with pytest.raises(ValueError, match="not synchronized"):
        release_versions.assert_synchronized(root, server)


def test_validate_release_requires_matching_prepared_dev_line(tmp_path):
    root = tmp_path / "pyproject.toml"
    server = tmp_path / "server.toml"
    root.write_text('[project]\nversion = "0.3.1.dev5"\n')
    server.write_text('dependencies = ["opensportslib==0.3.1.dev5"]\n')

    assert release_versions.validate_release("v0.3.1", root, server) == "0.3.1.dev5"
    with pytest.raises(ValueError, match="requires prepared version"):
        release_versions.validate_release("v0.3.2", root, server)


def test_prepare_development_starts_newer_feature_line(tmp_path):
    root = tmp_path / "pyproject.toml"
    server = tmp_path / "server.toml"
    root.write_text('[project]\nversion = "0.3.2.dev8"\n')
    server.write_text('dependencies = ["opensportslib==0.3.2.dev8"]\n')

    assert release_versions.prepare_development("v0.4.0", root, server) == "0.4.0.dev0"
    assert release_versions.assert_synchronized(root, server) == "0.4.0.dev0"
    with pytest.raises(ValueError, match="must be newer"):
        release_versions.prepare_development("v0.4.0", root, server)


def test_advance_development_moves_to_next_patch_only_from_matching_line(tmp_path):
    root = tmp_path / "pyproject.toml"
    server = tmp_path / "server.toml"
    root.write_text('[project]\nversion = "0.3.1.dev6"\n')
    server.write_text('dependencies = ["opensportslib==0.3.1.dev6"]\n')

    assert release_versions.advance_development("v0.3.1", root, server) == "0.3.2.dev0"
    assert release_versions.assert_synchronized(root, server) == "0.3.2.dev0"


def test_advance_development_refuses_a_different_release_line(tmp_path):
    root = tmp_path / "pyproject.toml"
    server = tmp_path / "server.toml"
    root.write_text('[project]\nversion = "0.4.0.dev0"\n')
    server.write_text('dependencies = ["opensportslib==0.4.0.dev0"]\n')

    with pytest.raises(ValueError, match="requires prepared version"):
        release_versions.advance_development("v0.3.1", root, server)
