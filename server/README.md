# OpenSportsLib Server

Async FastAPI + RQ backend for serving `opensportslib` inference from the `server/` project inside the OpenSportsLib repository.

## What this project does

- Exposes HTTP endpoints for a frontend to submit inference jobs
- Queues jobs in Redis using RQ
- Runs a separate worker process that keeps the active model loaded in-process for reuse across same-model jobs
- Supports:
  - VQA via `video_path + question`
  - VQA via direct `upload_file`
  - Classification via generated OSL JSON manifests
  - Localization via generated OSL JSON manifests
  - Session-aware API responses across all three tasks

## Monorepo installation

Imported from `OpenSportsLab/opensportslib-server` commit
`ee37ba1050dbb05747723ae53d6d4b13aa593378`. The main `opensportslib/` package
is unchanged. The server is maintained here as a separate installable project;
`pip install opensportslib` does not install or launch it.

From the **repository root**, use Python 3.12 or newer:

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install pip -e ./server
```

Alternatively, activate a Python 3.12 Conda environment and run
`python -m pip install -e ./server`. This installs the local server and its
pinned OpenSportsLib release from PyPI; it does not install the local library.
Use a fresh environment so an existing editable library installation does not
satisfy the dependency instead.


Run the helper to configure the selected model dependencies, generate standalone
configs, and create `.env`:

```bash
bash server/scripts/setup_env.sh
```

The helper supports uv virtual environments and Conda. It defaults to the Qwen
profile; select `OSL_VQA_DEP_PROFILE=xvars` for X-VARS or `none` to keep an
already provisioned runtime. PyTorch/model dependency setup still requires pip.
Redis is a separate service: install it with your system package manager or
`conda install -c conda-forge redis`, or use Docker Compose below. When Redis is
already running locally, the scripts reuse it.

Then start the server:

```bash
cd server
# Review .env: enable only the models you intend to serve.
./scripts/start_all.sh
```

All commands below run from `server/` unless stated otherwise. API and worker
must share the same runtime directory. `.env` loads explicitly from `server/`;
existing environment variables take precedence. Relative runtime/config paths
resolve from that directory, independent of the caller's working directory.

When migrating from the separate repository, copy only deliberate `.env`
settings and custom configs. Start with fresh runtime data and drain old jobs
before switching clients. Local credentials, logs, and queued jobs are not imported.

The server's OpenSportsLib dependency is pinned to the version declared in the
root `pyproject.toml` (currently `0.3.1.dev3`). Update the exact dependency in
`server/pyproject.toml` whenever the root project version changes; the server's
own package version is independent. Publish the matching library release to
PyPI before building the server image. Unpublished local-library changes are
not included in the image.

For installation through the requirements file, run from the repository root:

```bash
python -m pip install -r server/requirements.txt
```

## Configure models

Models are managed at runtime through the model registry. Configure
`OSL_MODEL_ADMIN_TOKEN` and register a Hugging Face or server-local model; no
API or worker restart is required:

```python
from opensportslib import RemoteModelRegistry

registry = RemoteModelRegistry("http://localhost:8000", admin_token="change-this-secret")
operation = registry.register_model(
    task_type="classification",
    huggingface_model_id="OpenSportsLab/OSL-cls-action-mvitv2",
)
registry.wait_for_operation(operation["operation_id"])
```

Local paths must be below `OSL_MODEL_ROOT`. A directory must contain
`config.yaml`; a weights file requires `config_path`. A local `model_id` may be
chosen by the user or omitted in favor of the generated ID returned by the API.

The fixed `OSL_*_MODEL_*`, `OSL_*_CONFIG_PATH`, and `OSL_*_WEIGHTS` entries are
deprecated compatibility bootstrap settings, imported only when Redis is empty.

`./scripts/setup_env.sh` creates `.env` from `.env.example` when it is missing. If you skip that helper, copy `.env.example` to `.env` and fill in the model config and weights you want to serve.

This project now generates flat standalone configs directly under `config/`:

- `config/classification_video.standalone.yaml`
- `config/localization_video_dali.standalone.yaml`
- `config/vqa_qwen3_vl_native.standalone.yaml`
- `config/vqa_qwen2_5_vl_native.standalone.yaml`
- `config/vqa_xvars.standalone.yaml`

These are fully merged standalone configs, so they do not depend on the
original OpenSportsLib `configs/...` folder structure.

Key environment entries are:

- `OSL_CLASSIFICATION_*` for classification
- `OSL_LOCALIZATION_*` for localization
- `OSL_JOB_TIMEOUT_SECONDS=7200` sets the maximum execution time for each queued RQ inference job. This is server-side and independent of the OpenSportsLib client's `remote_timeout`, which only controls how long the client waits for a response.
- `OSL_VQA_QWEN3_*` for the Qwen3 VL VQA model
- `OSL_VQA_QWEN25_*` for the Qwen2.5 VL VQA model
- `OSL_VQA_XVARS_*` for the X-VARS VQA model

Each model entry has:

- `..._MODEL_ENABLED`
- `..._MODEL_ID`
- `..._CONFIG_PATH`
- `..._WEIGHTS`

The checked-in defaults are portable and repo-relative:

- `OSL_RUNTIME_DIR=./runtime`
- `OSL_CLASSIFICATION_CONFIG_PATH=./config/classification_video.standalone.yaml`
- `OSL_LOCALIZATION_CONFIG_PATH=./config/localization_video_dali.standalone.yaml`
- `OSL_VQA_QWEN3_CONFIG_PATH=./config/vqa_qwen3_vl_native.standalone.yaml`
- `OSL_VQA_QWEN25_CONFIG_PATH=./config/vqa_qwen2_5_vl_native.standalone.yaml`
- `OSL_VQA_XVARS_CONFIG_PATH=./config/vqa_xvars.standalone.yaml`

The default `model_id` values are the Hugging Face repo IDs themselves:

- `OpenSportsLab/OSL-cls-action-mvitv2`
- `OpenSportsLab/OSL-loc-snbas-2025-e2e`
- `OpenSportsLab/OSL-VQA-XFOUL-qwen3-8B-VL-lora`
- `OpenSportsLab/OSL-VQA-XFOUL-qwen2.5-7B-VL-lora`
- `OpenSportsLab/OSL-VQA-XFOUL-XVARS-lora`

Pass `model_id` in the `/predict` request to select which one to use. The worker only keeps one active model loaded at a time, so it can switch models dynamically to reduce GPU RAM pressure.

## Current setup

With the current setup:

- Python env: any active Conda environment (for example, `conda activate osl`)
- OpenSportsLib package: `opensportslib==0.3.1.dev3` from PyPI
- Redis runs locally
- Configs are generated as flat standalone YAML files under `config/`
- `model_id` in API requests should be the Hugging Face repo ID
- `weights` in `.env` are also the Hugging Face repo IDs
- `start_all.sh` can predownload all listed Hugging Face model repos into cache
- worker execution mode defaults to `simple` so the same loaded model can stay resident in RAM/GPU memory across same-model jobs
- worker idle unload defaults to 10 minutes so unused GPU memory is eventually released
- Docker can optionally predownload all listed Hugging Face model repos into the mounted cache before worker startup

## Worker execution mode

Set `OSL_WORKER_EXECUTION_MODE` in `.env`:

- `simple`
  Keeps inference in the main worker process. This is the recommended mode for GPU serving because the active model can remain loaded across requests and sessions.
- `forking`
  Uses the default RQ worker process model. This isolates jobs more strongly, but reloads model state on each job and is not recommended for your current inference-server use case.

Idle GPU release is controlled by `OSL_WORKER_IDLE_UNLOAD_SECONDS`:

- default: `600`
- set to `0` to disable idle unloading
- when the worker stays idle longer than this threshold, it unloads the active model and clears CUDA cache

## VQA dependency profiles

OpenSportsLib uses different dependency override profiles for:

- Qwen VQA
- X-VARS VQA

Use `setup_env.sh` with:

```bash
OSL_VQA_DEP_PROFILE=qwen ./scripts/setup_env.sh
```

or:

```bash
OSL_VQA_DEP_PROFILE=xvars ./scripts/setup_env.sh
```

Only one profile should be active in the environment at a time.

## Start the API

```bash
./scripts/start_api.sh
```

## Start the worker

```bash
./scripts/start_worker.sh
```

## Start Redis

`./scripts/start_all.sh` supports both Redis management modes. It reuses a healthy Redis server already listening on the configured host and port; otherwise it starts the project-managed Redis binary.

If Redis is installed as a system service with `sudo apt install redis-server`, it is already running after:

```bash
sudo systemctl enable --now redis-server
```

You can then run `./scripts/start_all.sh`; it will reuse the system Redis service instead of starting a second instance.

If Redis is installed in your Conda environment, start the project-managed server with:

```bash
./scripts/start_redis.sh
```

The project-managed Redis script accepts an explicit port. The API and worker must use the same port in `OSL_REDIS_URL`:

```bash
REDIS_PORT=6380 OSL_REDIS_URL=redis://127.0.0.1:6380/0 ./scripts/start_all.sh
```

Use this when you want the project-managed Redis server to use a different port. If a healthy Redis service already occupies `6379`, `start_all.sh` reuses it automatically.

## API endpoints

- `GET /health`
- `GET /config-capabilities?task_type={task}&model_id={model}`
- `POST /predict`
- `GET /jobs/{job_id}`
- `GET /jobs/{job_id}/result`

Every successful `/predict` response now includes:

- `job_id`
- `session_id`

VQA supports follow-up questions on the same session. Classification and localization can also reuse the same `session_id`, but their behavior is different:

- no new input: return the latest successful cached result from that session
- new input: create a new job under the same session

`POST /predict` accepts both:

- `application/json` for `video_path` or `media_url`
- `multipart/form-data` for `upload_file`

Call `GET /config-capabilities` before sending request-specific inference
settings. Supported settings are carried in `task_options.config_overrides`:

```json
{
  "version": 1,
  "inference": {"max_new_tokens": 64, "temperature": 0.7}
}
```

The server validates these settings independently, records them with the job,
and restores the cached model configuration after every request. Device,
worker, output-path, dotted-path, and training overrides are not accepted.

## Example request

```json
{
  "task_type": "vqa",
  "video_path": "/abs/path/to/video.mp4",
  "question": "What card should be given?",
  "model_id": "OpenSportsLab/OSL-VQA-XFOUL-qwen2.5-7B-VL-lora"
}
```

## End-to-end usage

### 1. Setup

```bash
conda activate osl
cd /path/to/opensportslib-server
./scripts/setup_env.sh
```

### 2. Start services

```bash
./scripts/start_all.sh
```

By default this also predownloads all configured Hugging Face repos into cache
before starting Redis, API, and worker. To skip that step:

```bash
OSL_PREDOWNLOAD_ON_START=false ./scripts/start_all.sh
```

This same command works with the system Redis service installed with `sudo`; the existing Redis process is reused.

### 3. Check health

```bash
curl http://127.0.0.1:8000/health
```

### 4. Run VQA

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "task_type": "vqa",
    "model_id": "OpenSportsLab/OSL-VQA-XFOUL-qwen2.5-7B-VL-lora",
    "video_path": "/path/to/video.mp4",
    "question": "What is happening in this clip?"
  }'
```

Example response:

```json
{
  "job_id": "8c76c3dd-7d88-46e4-bde4-c4a6cb11aee8",
  "session_id": "50ca0608-0d90-4730-a4a0-0f9f74a09ecb",
  "status": "queued",
  "task_type": "vqa",
  "model_id": "OpenSportsLab/OSL-VQA-XFOUL-qwen2.5-7B-VL-lora",
  "created_at": "2026-08-13T10:00:00.000000Z"
}
```

### 4a. Run VQA with direct file upload

Use the same `/predict` endpoint with multipart form data and the file field name `upload_file`:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -F "task_type=vqa" \
  -F "model_id=OpenSportsLab/OSL-VQA-XFOUL-qwen2.5-7B-VL-lora" \
  -F "question=What is happening in this clip?" \
  -F "upload_file=@/path/to/video.mp4"
```

### 4c. Run a VQA follow-up on the same session

Use the `session_id` returned by the first VQA request. Do not send a new file or video path on the follow-up.

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "task_type": "vqa",
    "session_id": "50ca0608-0d90-4730-a4a0-0f9f74a09ecb",
    "question": "Was this a foul?"
  }'
```

### 4b. Run VQA with X-VARS

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "task_type": "vqa",
    "model_id": "OpenSportsLab/OSL-VQA-XFOUL-XVARS-lora",
    "video_path": "/path/to/video.mp4",
    "question": "What is happening in this clip?"
  }'
```

### 5. Run classification

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "task_type": "classification",
    "model_id": "OpenSportsLab/OSL-cls-action-mvitv2",
    "video_path": "/path/to/video.mp4"
  }'
```

Classification also returns a `session_id`.

If you send the same `session_id` again:

- with no new input, the API returns the latest successful cached result from that session
- with a new `video_path` or `upload_file`, the API creates a new job under the same session

### 6. Run localization

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "task_type": "localization",
    "model_id": "OpenSportsLab/OSL-loc-snbas-2025-e2e",
    "video_path": "/absolute/path/to/video.mp4"
  }'
```

Localization also returns a `session_id`.

If you send the same `session_id` again:

- with no new input, the API returns the latest successful cached result from that session
- with a new `video_path` or `upload_file`, the API creates a new job under the same session

### 7. Check job status

```bash
curl http://127.0.0.1:8000/jobs/<job_id>
```

The job status response includes `session_id` so the frontend can keep the request associated with the session.

### 8. Get result

```bash
curl http://127.0.0.1:8000/jobs/<job_id>/result
```

If a classification or localization request is repeated with the same `session_id` and no new input, `/predict` can return the cached result immediately. In that case the response includes:

```json
{
  "reused_result": true
}
```

Results are also saved on disk at:

```bash
runtime/results/<job_id>.json
```

### 9. Stop services

```bash
./scripts/stop_all.sh
```

`stop_all.sh` also clears server-generated runtime state so storage does not accumulate unnecessarily:

- uploaded temp files
- job metadata
- job result files
- session state files
- Redis runtime files under `runtime/redis`

While the server is running, uploaded media is kept for active sessions so VQA follow-up requests can reuse the same video. Expired sessions are cleaned automatically along with their associated runtime artifacts.

## Logs

- Redis: `logs/redis.log`
- API: `logs/api.log`
- Worker: `logs/worker.log`

## Session behavior

- Every accepted request creates or belongs to a session.
- Session TTL defaults to 30 minutes of idle time and is controlled by `OSL_SESSION_TTL_SECONDS`.
- VQA can reuse a session for follow-up questions on the same video.
- Classification and localization reuse the same session for workflow continuity.
- Classification and localization with no new input return the latest cached result from that session immediately.
- Classification and localization with new input create a new job under the same session.
- VQA follow-up requests must not send a new `upload_file`, `video_path`, or `media_url`.
- VQA follow-up requests create a new `job_id` under the same `session_id`.

## Docker

This project also includes a Docker deployment path with Conda already installed in the image.

This path is intended for the same backend behavior as the non-Docker setup:

- `/predict` supports JSON and `upload_file`
- successful requests return `job_id` and `session_id`
- worker uses `simple` execution mode unless overridden
- worker can keep the active model loaded and also unload it after `OSL_WORKER_IDLE_UNLOAD_SECONDS`

Files:

- `Dockerfile`
- `docker-compose.yml`

### What the image does

The Dockerfile is intentionally build-focused. Runtime defaults such as the
Conda `PATH`, Redis address, runtime directory, and standalone model-config
paths are set in `scripts/docker_entrypoint.sh`. Values supplied through
Compose, `docker run --env`, or `.env` take precedence over those defaults.

- starts from `continuumio/miniconda3`
- creates a Conda env named `osl`
- installs Python 3.12
- copies and installs only the server from the repository build context
- downloads the pinned OpenSportsLib release from PyPI
- generates the flat standalone configs under `/app/config`
- runs `opensportslib setup` for the selected VQA dependency profile when the container starts

This is intentional because the live container setup path matched the working host behavior better than running `opensportslib setup` during `docker build`.

### Important runtime note

For this server's request-driven inference flow, Docker only needs the runtime, logs, config, and Hugging Face cache mounts. Uploaded media and URLs do not need a dataset mount. A `video_path` must exist inside the worker container; add a read-only mount when using host files.

The Compose file mounts the Hugging Face cache like this:

```bash
${HOST_HF_CACHE:-./runtime/huggingface-cache}:/root/.cache/huggingface
```

### Docker environment values

The Docker launcher creates `.env` from `.env.example` automatically on its first run. Set or adjust these values in `.env` before building when needed:

- `OSL_DOCKER_VQA_DEP_PROFILE=qwen`
  Use `qwen`, `xvars`, or `none`
- `HOST_HF_CACHE=./runtime/huggingface-cache`
  Host Hugging Face cache path to reuse downloads

### GPU deployment

The single `docker-compose.yml` starts Redis, the API, and a worker with
`gpus: all`. The host must have a compatible NVIDIA driver and NVIDIA container
support. No additional Compose files are needed.

The default image uses `server/Dockerfile` and does not force the Spark-specific
CUDA wheel reinstall. For a Spark host, change the shared `build.dockerfile`
value in `docker-compose.yml` to `server/Dockerfile_spark`; that image enables
the existing CUDA 13.0 reinstall and architecture 12.1 settings.

### Build and run with Docker Compose

From `server/`:

```bash
./scripts/docker_compose.sh build
./scripts/docker_compose.sh up -d
```

Running `./scripts/docker_compose.sh` without arguments also creates `.env` when needed and starts the stack with `up -d`.

This starts:

- `redis`
- `api`
- `worker`

At container startup, the `api` and `worker` services will first run `opensportslib setup` for the selected `OSL_DOCKER_VQA_DEP_PROFILE`, then launch the service process.

Optional Docker predownload is also supported. In v1, this is intended to run on the `worker` service only so the shared mounted Hugging Face cache is filled once before the worker starts.

Set these in `.env` if you want that behavior:

- `OSL_DOCKER_PREDOWNLOAD_ON_START=true`
- `OSL_DOCKER_PREDOWNLOAD_WORKER_ONLY=true`

With that setting:

- `worker` predownloads all repos listed by `scripts/predownload_hf_assets.py`
- downloads go into the mounted `HOST_HF_CACHE`
- `api` skips the predownload step and still starts normally

If you want both `api` and `worker` to attempt predownload, set:

```bash
OSL_DOCKER_PREDOWNLOAD_WORKER_ONLY=false
```

The recommended v1 setting is to keep `OSL_DOCKER_PREDOWNLOAD_WORKER_ONLY=true`.

The entrypoint defaults to `qwen` when `OSL_DOCKER_VQA_DEP_PROFILE` is unset
or empty. This is a runtime setting, not a Docker build argument.

To use X-VARS, set `OSL_DOCKER_VQA_DEP_PROFILE=xvars` in `server/.env`, then
recreate the containers:

```bash
./scripts/docker_compose.sh up -d --force-recreate
```

Use `none` to skip dependency setup. Changing profiles does not require rebuilding
the image; recreating the containers applies the updated environment.

Changing Docker predownload settings also requires restarting the containers:

```bash
./scripts/docker_compose.sh down
./scripts/docker_compose.sh up -d
```

### Check service status

```bash
./scripts/docker_compose.sh ps
./scripts/docker_compose.sh logs -f api
./scripts/docker_compose.sh logs -f worker
./scripts/docker_compose.sh logs -f redis
```

After `./scripts/docker_compose.sh up -d`, check `./scripts/docker_compose.sh logs -f worker` to confirm `opensportslib setup` completed before sending inference requests.

If Docker predownload is enabled, the worker logs should also show that Hugging Face assets were downloaded before `opensportslib setup` began.

### Test the API

```bash
curl http://127.0.0.1:8000/health
```

You can then use the same `/predict`, `/jobs/{job_id}`, and `/jobs/{job_id}/result` commands documented earlier in this README.

### Stop Docker services

```bash
./scripts/docker_stop_all.sh
```

This stops the containers and clears runtime artifacts created by Docker. If you only want to stop containers without cleanup, use:

```bash
./scripts/docker_compose.sh down
```

If you have older `nobody:nogroup` runtime files from earlier Docker runs, `docker_stop_all.sh` is the preferred cleanup path because it performs the runtime wipe from inside Docker as root.

### Rebuild after code changes

```bash
./scripts/docker_compose.sh build
./scripts/docker_compose.sh up -d
```

### Model IDs in Docker requests

Use the same Hugging Face repo IDs as `model_id`:

- `OpenSportsLab/OSL-cls-action-mvitv2`
- `OpenSportsLab/OSL-loc-snbas-2025-e2e`
- `OpenSportsLab/OSL-VQA-XFOUL-qwen3-8B-VL-lora`
- `OpenSportsLab/OSL-VQA-XFOUL-qwen2.5-7B-VL-lora`

## Verification on the deployment host

From the repository root with the environment activated:

```bash
uv pip install pytest pytest-cov build
python -m pytest tests/test_*.py
python -m build
```

Inspect the resulting library wheel and source archive: neither should contain
`server/`. Start the API, Redis, and worker using the instructions above, then
check `/health` and submit one real inference job for classification, localization,
and VQA. Verify job polling and result retrieval, video and manifest uploads,
VQA session follow-ups, and configuration overrides with the current library
client. Exercise GPU inference on the intended hardware.

Local integration checks cover Python/shell syntax, YAML/TOML parsing, package
discovery, and the unchanged main-library source. Runtime, Docker, and model
verification are left to the deployment host; no passing runtime tests are claimed.
