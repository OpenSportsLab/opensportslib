import importlib.util
import json
from pathlib import Path


def queue_module():
    path = Path(__file__).parents[3] / "tools/reproduction/paper_queue.py"
    spec = importlib.util.spec_from_file_location("paper_queue", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_e2e_waits_for_all_tdeed_seeds_then_still_requires_verification(tmp_path, monkeypatch):
    queue = queue_module()
    (tmp_path / "tdeed.json").write_text(json.dumps({"experiments": [
        {"id": "main", "group": "released-main", "seeds": [1, 2]},
        {"id": "ablation", "group": "controlled-ablation", "seeds": [1]},
    ]}))
    entry = {"id": "e2e", "phase": "main", "seeds": [1], "canonical_config": "model.yaml"}
    (tmp_path / "e2e.json").write_text(json.dumps({"experiments": [entry]}))
    schedule = {
        "predecessor": {"name": "tdeed", "matrix": "tdeed.json",
                        "groups": ["released-main"], "output": "tdeed"},
        "campaign": {"name": "e2e", "matrix": "e2e.json", "phases": ["main"],
                     "output": "e2e", "verification": "verification.json"},
    }
    called = []
    monkeypatch.setattr(queue, "readiness", lambda *a: called.append(True) or {
        "ready": False, "errors": ["verification missing"]})
    for seed in [1, 2]:
        path = tmp_path / "tdeed/main" / f"seed-{seed}/results.json"
        path.parent.mkdir(parents=True)
        # A file without metrics is not a completed experiment.
        path.write_text(json.dumps({"seed": seed, "metrics": {}}))
    state = queue.queue_status(schedule, tmp_path)
    assert state["predecessor"]["completed"] == 0
    assert state["jobs"][0]["status"] == "waiting_for_tdeed_main"
    for seed in [1, 2]:
        path = tmp_path / "tdeed/main" / f"seed-{seed}/results.json"
        path.write_text(json.dumps({"seed": seed, "metrics": {"map": .5}}))
    state = queue.queue_status(schedule, tmp_path)
    assert state["predecessor"]["complete"]
    assert state["jobs"][0]["status"] == "blocked_prerequisites"
    assert called
    monkeypatch.setattr(queue, "readiness", lambda *a: {"ready": True, "errors": []})
    assert queue.queue_status(schedule, tmp_path)["jobs"][0]["status"] == "ready"


def test_main_queue_contains_24_paper_runs_and_four_bas_runs():
    import yaml
    queue = queue_module()
    root = Path(__file__).parents[3]
    schedule = yaml.safe_load((root / "reproduction/main-queue.yaml").read_text())
    jobs = queue.selected_jobs(schedule["campaign"], root)
    assert len(jobs) == 28
    assert sum(e["phase"] == "main" for e, seed in jobs) == 24
    assert sum(e["dataset"] == "soccernetball" for e, seed in jobs) == 4
    assert all((root / e["canonical_config"]).is_file() for e, seed in jobs)


def test_verification_is_invalidated_when_config_or_implementation_changes(tmp_path, monkeypatch):
    queue = queue_module()
    identity = {"config_sha256": "one", "implementation_sha256": {"model": "a"},
                "manifest_sha256": {"train": "b"}}
    monkeypatch.setattr(queue, "identity_and_data", lambda *a: (identity, []))
    entry = {"id": "model", "canonical_config": "model.yaml"}
    spec = {"verification": "verification.json"}
    assert not queue.readiness(entry, spec, tmp_path)["ready"]
    record = {"status": "passed", **identity, "checks": {c: True for c in queue.CHECKS}}
    (tmp_path / "verification.json").write_text(json.dumps({"experiments": {"model": record}}))
    assert queue.readiness(entry, spec, tmp_path)["ready"]
    identity["config_sha256"] = "two"
    assert not queue.readiness(entry, spec, tmp_path)["ready"]
