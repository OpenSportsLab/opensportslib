# OpenSportsLib inference server

The server is a FastAPI API backed by Redis and an RQ worker. It serves
classification, localization, and VQA models through HTTP and OpenSportsLib.

## Start the server

From the repository root, install the separate server project in Python 3.12+:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ./server
cd server
bash scripts/setup_env.sh
./scripts/start_all.sh
```

`start_all.sh` starts or reuses Redis, the API, and the worker. Docker Compose
is available with `./scripts/docker_compose.sh up -d`. A host `video_path` must
be visible inside the worker container; mount it read-only when using Docker.
See [server/README.md](../../server/README.md) for Docker, GPU, dependency,
and shutdown details.

Check readiness:

```bash
curl http://127.0.0.1:8000/health
```

Important settings include `OSL_REDIS_URL`, `OSL_RUNTIME_DIR`,
`OSL_MODEL_ROOT`, `OSL_MODEL_ADMIN_TOKEN`, `OSL_JOB_TIMEOUT_SECONDS`,
`OSL_MODEL_OPERATION_TIMEOUT_SECONDS`, `OSL_SESSION_TTL_SECONDS`,
`OSL_WORKER_EXECUTION_MODE`, and `OSL_WORKER_IDLE_UNLOAD_SECONDS`.

### Docker admin token setup

`server/.env.example` is only a template. Docker Compose loads the actual
`server/.env` file:

```bash
cd server
cp .env.example .env
```

Edit `server/.env` and set one private token assignment:

```env
OSL_MODEL_ADMIN_TOKEN=<your-private-admin-token>
```

Do not put a real token in `.env.example`, documentation, or Git. The token
shared during troubleshooting should be treated as exposed and rotated before
production use. Recreate the containers after changing it:

```bash
./scripts/docker_compose.sh down
./scripts/docker_compose.sh up -d --force-recreate
```

Verify that the API container received a token without printing its value:

```bash
docker compose exec api sh -c \
  'test -n "$OSL_MODEL_ADMIN_TOKEN" && echo "admin token is set" || echo "admin token is missing"'
```

Use the exact same private value in curl or Python:

```bash
curl -H "Authorization: Bearer <same-private-token>" \
  http://127.0.0.1:8000/models
```

```python
from opensportslib import RemoteModelRegistry

registry = RemoteModelRegistry(
    "http://10.64.74.111:8000",
    admin_token="<same-private-token>",
)
```

If authentication returns 401, check that you edited `.env` rather than only
`.env.example`, removed duplicate `OSL_MODEL_ADMIN_TOKEN` lines, recreated the
containers, connected to the expected host, included the `Bearer ` prefix, and
did not add quotes or trailing whitespace to the token.

## Model registry

Administration endpoints require `Authorization: Bearer $OSL_MODEL_ADMIN_TOKEN`.
Leave `OSL_MODEL_ADMIN_TOKEN` empty to disable administration. Public list and
status responses redact local filesystem paths.

### Register models

Hugging Face models use their repository ID as `model_id`:

```bash
curl -X POST http://127.0.0.1:8000/models \
  -H "Authorization: Bearer $OSL_MODEL_ADMIN_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"task_type":"classification","source":{"type":"huggingface","model_id":"OpenSportsLab/OSL-cls-action-mvitv2"}}'
```

Server-local models must be below `OSL_MODEL_ROOT`:

```bash
curl -X POST http://127.0.0.1:8000/models \
  -H "Authorization: Bearer $OSL_MODEL_ADMIN_TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"task_type":"localization","model_id":"football-model:v2","source":{"type":"local","weights_path":"/srv/models/football/model.pth","config_path":"/srv/models/football/config.yaml"}}'
```

A model directory discovers `config.yaml`; a weights file requires
`config_path`. Omitting a local ID generates `local:<name>-<hash>`. IDs may
contain letters, numbers, `.`, `_`, `-`, and `:`. Repeating the same
registration is idempotent; a different source for an existing ID returns 409.

Registration is asynchronous and returns an `operation_id`:

```bash
curl -H "Authorization: Bearer $OSL_MODEL_ADMIN_TOKEN" \
  http://127.0.0.1:8000/model-operations/<operation_id>
curl http://127.0.0.1:8000/models
curl 'http://127.0.0.1:8000/models/status?model_id=football-model:v2'
```

Model states are `registering`, `ready`, `failed`, and `unregistering`.
Operations are `queued`, `running`, `succeeded`, or `failed`; inference only
accepts `ready` models.

Set a task default and unregister a model:

```bash
curl -X PUT http://127.0.0.1:8000/models/defaults/classification \
  -H "Authorization: Bearer $OSL_MODEL_ADMIN_TOKEN" \
  -H 'Content-Type: application/json' -d '{"model_id":"OpenSportsLab/OSL-cls-action-mvitv2"}'
curl -X DELETE http://127.0.0.1:8000/models/football-model%3Av2 \
  -H "Authorization: Bearer $OSL_MODEL_ADMIN_TOKEN"
```

Poll the returned unregister operation before reusing the ID.

The Python administration client exposes the same lifecycle:

```python
from opensportslib import RemoteModelRegistry

r = RemoteModelRegistry("http://127.0.0.1:8000", admin_token="secret")
op = r.register_model(task_type="classification",
                      huggingface_model_id="OpenSportsLab/OSL-cls-action-mvitv2")
r.wait_for_operation(op["operation_id"])
print(r.list_models())
print(r.get_model("OpenSportsLab/OSL-cls-action-mvitv2"))
r.set_default("classification", "OpenSportsLab/OSL-cls-action-mvitv2")
remove = r.unregister_model("OpenSportsLab/OSL-cls-action-mvitv2")
r.wait_for_operation(remove["operation_id"])
```

`RemoteRegistryError` preserves HTTP status and structured detail. Connection
and polling timeouts raise `ConnectionError` and `TimeoutError`.

## Inference API

The Python task wrappers can be remote-only. When `remote` is configured, the
client does not need local `config` or `weights`; the registered server model
owns its configuration and checkpoint. `remote_model_id` is the server registry
ID and is independent of local `weights`.

```python
from opensportslib.apis import VQAModel

vqa = VQAModel(
    remote="http://10.64.74.111:8000",
    remote_model_id="OpenSportsLab/OSL-VQA-XFOUL-qwen3-8B-VL-lora",
)
```

The wrappers retain the `session_id` returned by the first real request as
`last_remote_session_id`. VQA questions reuse it automatically; classification
and localization calls without new input reuse the latest cached result. Use
`clear_remote_session()` to reset it or pass an explicit `session_id=`.

All requests use `POST /predict` and return a queued `job_id` and `session_id`.
Poll and retrieve results with:

```bash
curl http://127.0.0.1:8000/jobs/<job_id>
curl http://127.0.0.1:8000/jobs/<job_id>/result
```

If `model_id` is omitted, the ready task default is used. Discover supported
request-specific overrides first:

```bash
curl 'http://127.0.0.1:8000/config-capabilities?task_type=vqa&model_id=OpenSportsLab/OSL-VQA-XFOUL-qwen3-8B-VL-lora'
```

Only advertised settings in `task_options.config_overrides` are accepted.

### Single video

Use a server-readable path or configured URL:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"task_type":"classification","model_id":"OpenSportsLab/OSL-cls-action-mvitv2","video_path":"/data/clip.mp4"}'
```

VQA also requires `question`; `media_url` can replace `video_path` where URL
retrieval is configured. For an uploaded file:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F task_type=vqa -F model_id=OpenSportsLab/OSL-VQA-XFOUL-qwen3-8B-VL-lora \
  -F question='What happened?' -F upload_file=@clip.mp4
```

The Python wrappers upload local media when `remote` is set:

```python
from opensportslib.apis import ClassificationModel

m = ClassificationModel(config="classification.yaml",
    remote="http://127.0.0.1:8000",
    remote_model_id="OpenSportsLab/OSL-cls-action-mvitv2")
predictions = m.infer(video_path="clip.mp4")
```

For direct remote VQA, provide both the video and question; a follow-up uses
only the returned session ID:

```python
from opensportslib.apis import VQAModel

vqa = VQAModel(config="vqa.yaml", remote="http://127.0.0.1:8000",
               remote_model_id="OpenSportsLab/OSL-VQA-XFOUL-qwen3-8B-VL-lora")
answer = vqa.infer(video_path="clip.mp4", question="Was this a foul?")
follow_up = vqa.infer(question="What card should be given?", session_id="<session_id>")
```

### Entire test set

Send an OSL JSON manifest and a ZIP containing every referenced media file:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F task_type=classification \
  -F model_id=OpenSportsLab/OSL-cls-action-mvitv2 \
  -F test_set_file=@test.json -F media_archive=@media.zip
```

```python
predictions = m.infer(test_set="test.json", remote_mode="full_test_set")
```

Manifest/archive uploads cannot be combined with a single-video input or an
existing session.

### One job per sample

```python
predictions = m.infer(test_set="test.json", remote_mode="per_sample")
failures = m.last_remote_failures
```

Each sample and its referenced media are submitted independently. Successful
predictions are returned after individual failures, which are recorded in
`last_remote_failures`. `full_test_set` uses one upload/job; `per_sample`
provides isolation and partial progress.

With raw HTTP, repeat the multipart request for each sample and its matching
media archive (the shell loop is application-specific):

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F task_type=classification -F model_id=OpenSportsLab/OSL-cls-action-mvitv2 \
  -F test_set_file=@sample-001.json -F media_archive=@sample-001-media.zip
```

## Sessions and VQA follow-ups

```bash
curl -X POST http://127.0.0.1:8000/predict -H 'Content-Type: application/json' \
  -d '{"task_type":"vqa","model_id":"OpenSportsLab/OSL-VQA-XFOUL-qwen3-8B-VL-lora","video_path":"clip.mp4","question":"Was this a foul?"}'
curl -X POST http://127.0.0.1:8000/predict -H 'Content-Type: application/json' \
  -d '{"task_type":"vqa","session_id":"<session_id>","question":"What card should be given?"}'
```

Classification and localization may reuse a session with no new input to return
the latest successful cached result. The OpenSportsLib wrappers start a new
session automatically when a new video is supplied; direct HTTP callers may
explicitly reuse a session if desired. VQA follow-ups must not include a file, path, or URL. Sessions expire
after `OSL_SESSION_TTL_SECONDS` and then return 410.

Common errors are 401 invalid admin token, 404 unknown model/job/session, 409
model not ready or result unavailable, 422 invalid input/override, and 415
unsupported content type.
