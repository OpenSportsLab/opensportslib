from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SERVER_ROOT = REPOSITORY_ROOT / "server"
sys.path.insert(0, str(SERVER_ROOT))

from opensportslib.apis.base_task_model import BaseTaskModel  # noqa: E402


def test_remote_session_state_is_captured_and_clearable():
    model = type(
        "RemoteState",
        (),
        {"_remember_remote_session": BaseTaskModel._remember_remote_session},
    )()
    model.remote = "http://server"
    model.remote_model_id = "model"
    model.remote_session_id = None

    BaseTaskModel._remember_remote_session(model, {"session_id": "session-1"})
    assert BaseTaskModel.last_remote_session_id.fget(model) == "session-1"

    BaseTaskModel.clear_remote_session(model)
    assert BaseTaskModel.last_remote_session_id.fget(model) is None


def test_submit_session_inference_sends_session_without_media(monkeypatch):
    model = type(
        "RemoteState",
        (),
        {"_remember_remote_session": BaseTaskModel._remember_remote_session},
    )()
    model.remote = "http://server"
    model.remote_model_id = "model"
    model.remote_session_id = None
    captured = {}

    def post(path, *, fields, files):
        captured.update(path=path, fields=fields, files=files)
        return {"job_id": "job-1", "session_id": "session-1", "status": "queued"}

    monkeypatch.setattr(model, "_post_multipart", post, raising=False)
    response = BaseTaskModel.submit_session_inference(
        model,
        task_type="vqa",
        session_id="session-0",
        question="What happened?",
    )

    assert response["session_id"] == "session-1"
    assert BaseTaskModel.last_remote_session_id.fget(model) == "session-1"
    assert captured["path"] == "/predict"
    assert captured["fields"]["session_id"] == "session-0"
    assert captured["files"] == {}


def test_manifest_upload_bundles_tracab_metadata_without_rewriting_source(tmp_path):
    import copy
    import json
    import zipfile
    from tests.helpers.tracab import write_tracab

    descriptor = write_tracab(tmp_path / 'match.dat', sidecar=True)
    descriptor['path'] = 'match.dat'
    descriptor['metadata_path'] = 'match.metadata.json'
    for key in ('tf05_path', 'tf08_path'):
        descriptor[key] = str(Path(descriptor[key]).relative_to(tmp_path))
    payload = {'data': [{'id': 'match', 'inputs': [descriptor]}]}
    before = copy.deepcopy(payload)
    manifest, archive = BaseTaskModel._stage_manifest_payload(None, payload, tmp_path, tmp_path / 'upload')
    assert payload == before
    staged = json.loads(manifest.read_text())['data'][0]['inputs'][0]
    assert staged['metadata'] == descriptor['metadata']
    with zipfile.ZipFile(archive) as bundle:
        assert len(bundle.namelist()) == 4
        for key in ('path', 'metadata_path', 'tf05_path', 'tf08_path'):
            assert bundle.read(staged[key]) == (tmp_path / descriptor[key]).read_bytes()
