# Frozen video-to-skeleton preprocessing

OSL can feed COCO-17 player skeletons extracted from a video into its existing
SoccerNet-GAR graph architecture. The pose and person detector are frozen ONNX
sessions. The action encoder, temporal neck/head, and classifier remain trainable.

The initial candidate is YOLOX-m (640×640) plus RTMPose-s (192×256). RTMPose-m,
YOLOX-tiny, and one-stage RTMO-s are selectable benchmark alternatives. These
are candidate presets, **not GPU-validated production defaults**. No throughput,
quality, or real-time claim has been established for public tennis or ShotQuality.

## Optional installation

Install OSL and its optional PyG support first. In an existing CUDA 12/13 PyTorch
and cuDNN 9 environment, run:

```bash
opensportslib setup --pose
```

Pose-only setup preserves the current Torch installation. It installs
`rtmlib==0.0.16` without transitive dependencies and `scipy>=1.13,<2`, selecting
`onnxruntime-gpu==1.20.2` for CUDA 12 or `==1.31.0` for CUDA 13. These profiles
are version-pinned candidates; GPU verification is still required. CUDA 12 on
ARM64 requires a separately installed platform-supplied CUDA ONNX Runtime wheel;
the historical CUDA 12 pin has no published ARM64 wheel. Setup checks this before
installing packages. RTMLib's
metadata asks for CPU ONNX Runtime and two overlapping OpenCV distributions;
OSL deliberately uses the existing OpenCV/numpy/tqdm and GPU runtime instead.
Consequently `pip check` can report RTMLib's unused distribution requirements.
Do not install CPU `onnxruntime` beside `onnxruntime-gpu`. Use a dedicated pose
environment if another application requires that CPU distribution.

The backend checks the CUDA provider and refuses silent CPU fallback. CPU decode,
OpenCV crops, NumPy postprocessing, and host/device transfers still contribute to
end-to-end runtime. No MMCV, MMDetection, or MMPose runtime is needed. Cached
training and all existing OSL workflows do not import RTMLib or ONNX Runtime.

References: [RTMLib code/license](https://github.com/Tau-J/rtmlib),
[model definitions](https://github.com/Tau-J/rtmlib/blob/main/rtmlib/tools/solution/body.py),
[ONNX Runtime CUDA requirements](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).

## Inference

Use `examples/tennis/classification_skeleton.yaml` for one stroke clip or
`examples/tennis/localization_skeleton.yaml` for a broadcast. Set the action
classes and load a tennis skeleton action checkpoint; RGB and soccer checkpoints
are not interchangeable with the 73-feature skeleton model.

```python
from opensportslib.apis.classification import ClassificationModel
from opensportslib.apis.localization import LocalizationModel

classifier = ClassificationModel(config="examples/tennis/classification_skeleton.yaml")
prediction = classifier.infer(video_path="stroke.mp4", weights="classification.pt", use_wandb=False)

spotter = LocalizationModel(config="examples/tennis/localization_skeleton.yaml")
prediction = spotter.infer(video_path="match.mp4", weights="spotting.pt", use_wandb=False)
spotter.save_predictions(output_path="events.json", predictions=prediction)

stream = spotter.infer_stream(video_path="match.mp4", weights="spotting.pt")
try:
    for batch in stream:
        consume(batch)  # OSL localization JSON with finalized event batches
finally:
    stream.close()      # also close when stopping early
```

`infer_stream` is local-only. It returns completed predictions, not provisional
updates. No events are emitted for background-only batches. Do not concurrently
train, replace weights, or edit configuration on a model with an active iterator.
Each video receives independent association/window state. Predictions contain
class events; actor IDs remain in skeleton caches/overlays.

Classification samples the configured number of frames uniformly over the source
clip (respecting configured start/end bounds). Each view has independent tracking
state; the model averages per-view clip embeddings before classification. Training
keeps the existing loader's view-selection policy. Spotting keeps the existing
video loader's random/event-focused training windows and evaluation windows.

The loaders return unnormalized, uncropped uint8 RGB with exact source frame
indices, PTS/time base, timestamps, geometry and padding masks under `video_meta`.
The non-DALI raw path uses PyAV inside the existing video dataset interface for
source timestamps; ordinary RGB decoding/transforms retain their existing behavior.
Only selected images are retained. Deterministic resize is inverted before graph
features are built. Spatial/view/time padding never becomes an observation.
Image augmentation, normalization and mixup before pose are rejected explicitly.
Repeated frames in short uniform samples reuse observations without advancing tracks.

Training resets association for each sampled clip/view. Continuous inference uses
an explicit model-owned session across decoded chunks, resetting at cuts or video
boundaries. Detection sees the loader-selected frames; it cannot detect cuts in
frames the loader discarded. The two reset policies intentionally have different
initialization context and must not share cache entries.

Spotting uses 64-frame windows with 32-frame overlap by default. The existing
integer-stride sampling policy determines the effective sample rate, which can
differ from requested `extract_fps`. Each shot tail is masked, and temporal heads
run separately on valid shot segments, including clips with left padding. Cuts
flush old-shot scores and NMS context. The GRU is **bidirectional**: window
completion, overlap finalization and symmetric NMS introduce lookahead. A
64-sample window at 25 Hz spans about 2.56 seconds; actual event latency also
includes finalization policy and processing time. This is delayed spotting.

## Model configuration and training

Keep `DATA.inputs.video.modality: video`, `representation: raw`, and ordinary
video annotations. Define frozen extraction under the model:

```yaml
MODEL:
  components:
    pose_tracking:
      kind: preprocessor
      source: {provider: opensportslib, registry: preprocessor, name: pose_tracking}
      params:
        backend: rtmlib
        detector: yolox_m
        pose: rtmpose_s
        max_players: 2
        selection: tennis_conservative
        cache: {mode: off}
  # Add pose_tracking -> tracking_encoder before the existing topology.
IO:
  inputs: {video: pose_tracking, label: task_head}
```

The frontend declares its raw input requirements and 73-feature graph output.
It has no optimizer parameters and stays frozen under `model.train()`. Action
checkpoints contain graph/neck/head weights; pose artifacts remain external.
Runtime sessions are created lazily in the model process, never in workers.

Use `VideoDataset` through the existing classification builder. For spotting,
use `VideoGameWithOpencv` for train/valid and `VideoGameWithOpencvVideo` for
full-video evaluation. Set split source/annotation paths and class names, then:

```python
from opensportslib.apis import ClassificationModel, LocalizationModel

classifier = ClassificationModel(config="examples/tennis/classification_skeleton.yaml")
classifier.train(use_wandb=False)
spotter = LocalizationModel(config="examples/tennis/localization_skeleton.yaml")
spotter.train(use_wandb=False)
```

Annotations retain `inputs: [{type: video, path: clip.mp4}]`. Classification
records use `labels.action.label`; spotting records use `events` with `position_ms`.
For multiview classification, set `params.view_type: multi` and supply the same
video view records accepted by the existing loader. Player identities do not
associate across views.

The `_dali.yaml` tennis examples enable both tasks with optional DALI 2.2+.
The inspected CUDA 12 profile is `nvidia-dali-cuda120==2.2.0`; install the matching
DALI distribution for the host CUDA version and verify it in the release tier.
Classification supports one frontend per DDP rank. The legacy E2E DataParallel
mode is rejected for pose models; use one GPU for each spotting model process.
The internal adapter consumes the existing torch sampler, including weighting,
distributed sharding and drop-last behavior. GPU decoding runs in the training
process with bounded request readers and no decoder pool. Training rebuilds
readers per request; chronological streaming keeps one reader for the video.
A sequential PyAV clock pass preserves original PTS/time bases for DALI; these costs are included in benchmarks. It is not a promised
speed improvement. RTMLib needs host arrays, so GPU-decoded frames transfer back
to the host individually. The DALI path supports CFR only and rejects unsupported
sampling/clocks rather than silently changing backend. Non-DALI supports VFR.

## Exact-clip caches and explicit extraction

The frontend supports `cache.mode: off` (default), `read`, and `read_write`, with
`cache.path` naming a cache root. `read_write` explicitly opts into immutable
cache publication during training/inference; `read` fails on a missing exact
request. With pinned artifact SHA-256 values, reading a hit needs no RTMLib/ORT
runtime or CUDA frontend. Action weights do not invalidate extraction caches.

Warm caches explicitly through the existing configured video loaders:

```bash
python -m opensportslib.preprocessing warm-cache \
  --config examples/tennis/classification_skeleton.yaml \
  --split train --output data/tennis/clip-cache
python -m opensportslib.preprocessing warm-cache \
  --config examples/tennis/localization_skeleton.yaml \
  --split train --max-clips 100 --output data/tennis/clip-cache
```

Warm-up enumerates classification samples and seeds a sample of the existing
random spotting windows. Random windows in later epochs can miss this set;
use `read_write` to grow the cache explicitly, or `read` only with reproducible,
fully warmed requests. Annotations always continue to reference video files.
Logical intervals use the existing interval expansion and event-rebasing rules;
streaming resets at interval boundaries.

Continuous pre-extraction remains useful for visualization and diagnostics:

```bash
python -m opensportslib.preprocessing extract \
  --config examples/tennis/localization_skeleton.yaml \
  --video match.mp4 --output data/tennis/continuous/match
python -m opensportslib.preprocessing overlay \
  --video match.mp4 --cache data/tennis/continuous/match --output review.mp4
```

These continuous caches cannot substitute for independent training clips. The
model's streaming session bypasses clip caching to preserve its prior state.
Ordinary inference with caching off writes neither observations nor predictions;
use `save_predictions` explicitly. Missing model weights can still be downloaded.

**BREAKING CHANGE — experimental skeleton dataset/config migration:** remove
`SkeletonClassification`/`SkeletonSpotting` split types, `skeleton_cache` training
inputs and `representation: skeleton_graph`. Move `params.preprocessing` into
`MODEL.components.pose_tracking.params`, add the topology edge and IO mapping,
remove `params.objects` for this route, and use ordinary video inputs/types.
Old continuous caches are not reused as exact-clip caches; rebuild them with
`warm-cache` or opt into `read_write`. Existing RGB and soccer tracking configs
retain their contracts.

## Representation and identity

A canonical observation contains `[P,17,2]` float32 image-pixel joints,
`[P,17]` raw confidence and boolean joint validity, `[P,4]` xyxy boxes, player
validity, track IDs, shot ID, image dimensions, near/far roles, source frame/PTS/time
base, video-relative timestamp, and sampling-grid timestamp. Timestamps are
relative to the first decoded presentation timestamp; original PTS are retained.
Continuous sampling emits at most one observation per decoded frame. Clip
sampling may repeat a source observation, without re-running pose or advancing
association. Neither path fabricates intermediate poses. Missing/non-monotonic PTS
raise errors. Gaps over one second reset shot state. No joint interpolation is used.

Each player node has 73 float32 features: normalized box center/size, ordered
`(body_x, body_y, confidence, valid)` for 17 joints, and player validity. Body
coordinates subtract box center and divide by maximum box dimension. Invalid
coordinates become zero while confidence/masks remain explicit. Absent nodes are
excluded from graph computation/pooling; empty frames produce zero embeddings.
GIN uses LayerNorm for skeleton inputs so one visible player is valid in training.
Graph IDs/explicit graph count preserve trailing empty frames.

Tracks retain their slot within a shot. Missing players have ID -1 in observations;
internal tracks survive 0.5 seconds without reporting predicted joints as observed.
Reacquisition after expiry uses a new ID. IDs do not repeat across cuts in a video;
there is no promise of identifying a named athlete across shots.

Conservative tennis selection fits white court lines to a trapezoid, requires
stable geometry, and filters people by court-relative foot position and scale.
Ambiguous candidates are masked. This untrained heuristic can reject valid courts
under unusual lighting/angles and cannot reliably distinguish wide-court replays.
A hard-cut threshold can miss dissolves or mistake rapid motion for a cut. Inspect
overlays and quantify coverage/contamination before deployment. `selection:
all_people` is an explicit diagnostic/general-sport policy; it does not distinguish
competitors from officials.

## Cache integrity and model provenance

Caches are immutable directories containing 256-observation chunks of separate
NumPy arrays and a JSON manifest. Readers mmap one chunk at a time. Completed
caches publish by directory rename; failures remove temporary data. Existing
caches are not overwritten and incomplete runs restart from the beginning.

Identity includes source-video SHA-256, schema/sampler/selector/tracker versions,
extraction settings, selected indices/times, resize geometry, decoder profile,
clip-reset policy and resolved model checksums. Action-model weights are excluded.
New extraction settings or clip requests produce different cache entries.
The cache stores observations, not augmented tensors or graph objects.

Named model presets contain versioned upstream artifact URLs. The resolved ONNX
SHA-256 is recorded at initialization. Pin deployments using:

```yaml
MODEL:
  components:
    pose_tracking:
      params:
        artifacts:
          rtmpose_s:
            path: /models/rtmpose-s.onnx
            sha256: YOUR_VERIFIED_SHA256
```

RTMLib/MMPose/YOLOX publish Apache-2.0 code licenses. The
[RTMPose mirror](https://huggingface.co/Tau-J/RTMPose) declares Apache-2.0 for
models, but exact exported-weight provenance must be reviewed before redistribution.
Checksums attest artifact identity, not license clearance. No model binaries are
bundled. Preserve upstream copyright/NOTICE files when distributing dependencies
or approved weights. Ultralytics is not installed or required.

## Public tennis benchmark and release acceptance

On a CUDA machine, run an isolated GPU benchmark for each candidate:

```bash
python -m opensportslib.preprocessing benchmark \
  --config examples/tennis/localization_skeleton.yaml \
  --video public-tennis.mp4 --output benchmark-s.json --repeats 3
```

Repeat with `classification_skeleton.yaml` for classification; add
`--view second-camera.mp4` for multiview. Repeat with the `_dali.yaml` examples
and all candidate pose presets. First-result timing is clip completion for
classification and first emitted event batch for spotting. Classification's
model-forward stage includes frontend work; pose stage timings overlap it.

Set the action checkpoint in `MODEL.load.checkpoint_path`. Without a checkpoint,
the report measures randomly initialized action computation and establishes no
action quality. Use separate configs for RTMPose-m, YOLOX-tiny, and RTMO-s.
Benchmark outputs include artifact checksums, software/hardware, startup, warm
source/sampled throughput, first event-batch timing, finalization latency,
lookahead, detection/pose/action timing, RSS, and polled total GPU memory. GPU
memory includes other processes and 100ms polling can miss instantaneous peaks.
Post-extraction timing excludes the current frame's frontend work; the full-video
throughput includes decode, pose, tracking, graph construction, and action inference.

Prepare a version-pinned public manifest with documented rights and source-match
IDs. Do not infer dataset/publication permission from video availability. Use
existing public tennis release fixtures for baseline labels, supplementing them
with continuous broadcasts where they lack cuts/officials. Keep source matches
separate across train/validation/test. Review far-player joints, fast swings,
occlusions, missing players, close-ups/replays, and hard cuts. Annotate competitor
identity and visible joints on a small review subset to measure selection errors,
ID switches/fragmentation, reacquisition, and joint error. Automatic counters are
not ground-truth quality measurements.

For action evaluation, train the existing action models with optional clip caching and
report classification accuracy/balanced accuracy/macro-F1 and spotting mAP/F1
per class. Include bounce classes where present and report them separately;
body-only inputs may not identify ball bounces well. Compare cache/live inputs
and predictions before measuring accuracy. Run short and repeated long sources to
check that streaming RSS/VRAM plateau; complete result objects grow with event count.

Required contribution check: `bash scripts/run_tests.sh`. Fast tests use synthetic
backend outputs and real action models. Actual GPU/provider/weight checks belong
to opt-in release verification. Numerical performance and domain quality targets
remain unset until this public pilot has run. ShotQuality validation follows it.
