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
