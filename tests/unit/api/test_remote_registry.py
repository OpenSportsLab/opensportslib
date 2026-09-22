import json
from unittest.mock import patch

import pytest

from opensportslib import RemoteModelRegistry, RemoteRegistryError


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


def test_register_huggingface_uses_one_registration_method():
    client = RemoteModelRegistry("http://server")
    with patch("opensportslib.remote_registry.request.urlopen") as urlopen:
        urlopen.return_value = FakeResponse({"model_id": "org/model", "status": "registering"})
        result = client.register_model(task_type="classification", huggingface_model_id="org/model")

    outgoing = urlopen.call_args.args[0]
    assert json.loads(outgoing.data) == {
        "task_type": "classification",
        "source": {"type": "huggingface", "model_id": "org/model"},
    }
    assert "Authorization" not in outgoing.headers
    assert result["model_id"] == "org/model"


def test_register_local_supports_custom_or_generated_model_id():
    client = RemoteModelRegistry("http://server", api_key="secret")
    with patch("opensportslib.remote_registry.request.urlopen") as urlopen:
        urlopen.return_value = FakeResponse({"model_id": "custom", "status": "registering"})
        client.register_model(
            task_type="localization",
            model_id="custom",
            weights_path="/models/custom/model.pth",
            config_path="/models/custom/config.yaml",
        )

    payload = json.loads(urlopen.call_args.args[0].data)
    assert payload["model_id"] == "custom"
    assert payload["source"] == {
        "type": "local",
        "weights_path": "/models/custom/model.pth",
        "config_path": "/models/custom/config.yaml",
    }
    assert urlopen.call_args.args[0].headers["Authorization"] == "Bearer secret"


def test_huggingface_request_token_is_sent_only_in_registration_payload():
    client = RemoteModelRegistry("http://server")
    with patch("opensportslib.remote_registry.request.urlopen") as urlopen:
        urlopen.return_value = FakeResponse({"model_id": "org/private", "status": "registering"})
        client.register_model(
            task_type="classification", huggingface_model_id="org/private", hf_token="hf_private"
        )
    outgoing = urlopen.call_args.args[0]
    assert json.loads(outgoing.data)["source"]["hf_token"] == "hf_private"
    assert "Authorization" not in outgoing.headers


def test_unregister_sends_hf_token_in_header():
    client = RemoteModelRegistry("http://server")
    with patch("opensportslib.remote_registry.request.urlopen") as urlopen:
        urlopen.return_value = FakeResponse({"model_id": "org/private", "status": "unregistering"})
        client.unregister_model("org/private", hf_token="hf_private")
    outgoing = urlopen.call_args.args[0]
    assert outgoing.headers["X-hf-token"] == "hf_private"
    assert outgoing.data is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"huggingface_model_id": "org/model", "weights_path": "/models/model"},
    ],
)
def test_register_model_requires_exactly_one_source(kwargs):
    client = RemoteModelRegistry("http://server", "secret")
    with pytest.raises(ValueError, match="exactly one"):
        client.register_model(task_type="classification", **kwargs)


def test_huggingface_model_cannot_be_renamed():
    client = RemoteModelRegistry("http://server", "secret")
    with pytest.raises(ValueError, match="repository ID"):
        client.register_model(
            task_type="classification",
            huggingface_model_id="org/model",
            model_id="alias",
        )


def test_http_errors_preserve_structured_detail():
    from urllib.error import HTTPError
    from io import BytesIO

    client = RemoteModelRegistry("http://server", "secret")
    response = HTTPError(
        "http://server/models", 409, "Conflict", {},
        BytesIO(json.dumps({"detail": {"code": "MODEL_NOT_READY"}}).encode()),
    )
    with patch("opensportslib.remote_registry.request.urlopen", side_effect=response):
        with pytest.raises(RemoteRegistryError) as raised:
            client.get_model("model")
    assert raised.value.status_code == 409
    assert raised.value.detail == {"code": "MODEL_NOT_READY"}


def test_reconcile_runtime_defaults_to_dry_run_and_authenticates():
    client = RemoteModelRegistry("http://server", "secret")
    with patch("opensportslib.remote_registry.request.urlopen") as urlopen:
        urlopen.return_value = FakeResponse({"inspected": 2, "active": 1})
        result = client.reconcile_runtime()

    outgoing = urlopen.call_args.args[0]
    assert json.loads(outgoing.data) == {"dry_run": True, "include_active": False}
    assert outgoing.headers["Authorization"] == "Bearer secret"
    assert result["active"] == 1


def test_reconcile_runtime_can_apply_and_include_active():
    client = RemoteModelRegistry("http://server", "secret")
    with patch("opensportslib.remote_registry.request.urlopen") as urlopen:
        urlopen.return_value = FakeResponse({"recovered": 1})
        client.reconcile_runtime(dry_run=False, include_active=True)

    outgoing = urlopen.call_args.args[0]
    assert json.loads(outgoing.data) == {"dry_run": False, "include_active": True}
