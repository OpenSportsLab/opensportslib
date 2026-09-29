import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).parents[3]
SERVER_ROOT = REPO_ROOT / "server"
SERVERCTL = SERVER_ROOT / "scripts" / "serverctl"


def test_serverctl_help_lists_consolidated_commands():
    result = subprocess.run([str(SERVERCTL), "--help"], check=True, capture_output=True, text=True)
    for command in ("setup", "start", "stop", "status", "clean", "reset", "reconcile", "docker"):
        assert command in result.stdout


def test_compose_keeps_internal_ports_and_parameterizes_host_ports():
    compose = (SERVER_ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert '"${OSL_HOST_REDIS_PORT:-6379}:6379"' in compose
    assert '"${OSL_HOST_API_PORT:-8000}:8000"' in compose


def test_serverctl_contains_lifecycle_contracts():
    script = SERVERCTL.read_text(encoding="utf-8")
    assert "find_free_port" in script
    assert "redis_healthy" in script
    assert "service-endpoints.env" in script
    assert '"worker_alive":true' in script
    assert "Hugging Face cache preserved" in script
    assert "Reset requires interactive confirmation or --yes" in script
    assert "Factory reset complete" in script


def test_removed_script_names_are_not_referenced():
    removed = (
        "clean_runtime.sh", "docker_compose.sh", "docker_stop_all.sh", "download_all_weights.sh",
        "port_utils.sh", "predownload_hf_assets.py", "reconcile_runtime.py", "setup_env.sh",
        "start_all.sh", "start_api.sh", "start_docker.sh", "start_redis.sh", "start_worker.sh", "stop_all.sh",
    )
    paths = [REPO_ROOT / "README.md", SERVER_ROOT / "README.md", *list((REPO_ROOT / "docs").rglob("*.md"))]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert not any(name in text for name in removed), path


def test_dockerfiles_use_retained_build_scripts():
    for name in ("Dockerfile", "Dockerfile_spark"):
        dockerfile = (SERVER_ROOT / name).read_text(encoding="utf-8")
        assert "/app/scripts/generate_flat_configs.py" in dockerfile
        assert 'ENTRYPOINT ["/app/scripts/docker_entrypoint.sh"]' in dockerfile
        assert "chmod +x /app/scripts/serverctl /app/scripts/docker_entrypoint.sh" in dockerfile
