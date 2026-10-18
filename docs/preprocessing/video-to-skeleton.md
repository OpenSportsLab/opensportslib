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

Classification uses timestamp-uniform sampling and two bounded-memory decode
passes to obtain exact clip endpoints. Pose extraction occurs only on the second
pass. Spotting uses one decode pass, 25 sampled frames/s, 64-frame windows, and
32-frame overlap. Each shot tail is padded but masked from temporal computation
and training loss. No window crosses a detected camera cut. Cuts immediately flush old-shot
scores and shot-local NMS context.

The GRU is **bidirectional**. Window completion, overlap finalization, and
symmetric NMS all introduce lookahead. A 64-sample window at 25 Hz spans about
2.56 seconds; this is not a fixed event latency. Sparse timestamps and scheduling
also affect finalization. Benchmark lookahead separately from wall-clock work.
Fully causal spotting requires a different temporal contract and is deferred.

## Explicit extraction, cache, and training

```bash
python -m opensportslib.preprocessing extract \
  --config examples/tennis/localization_skeleton.yaml \
  --video match.mp4 --output data/tennis/cache/match
python -m opensportslib.preprocessing overlay \
  --video match.mp4 --cache data/tennis/cache/match --output skeleton-review.mp4
```

Python users can call `extract(video_path, destination, settings)` or use
`SkeletonSession.push/finish/close` with incoming BGR frames, increasing timestamps,
source PTS, and time base. Only the explicit extraction operation writes skeleton
caches. Inference may download missing **model weights** into its model directory;
it does not write skeleton or prediction files.

Training annotations reference extracted directories:

```json
{
  "version": "2.0",
  "labels": {"action": {"type": "single_label", "labels": ["swing"]}},
  "data": [{
    "id": "match_01",
    "inputs": [{"type": "skeleton_cache", "path": "cache/match_01"}],
    "events": [{"head": "action", "label": "swing", "position_ms": 1200}]
  }]
}
```

For classification, use a stroke cache per record and replace `events` with
`"labels": {"action": {"label": "swing"}}`. Source paths are relative to the
split's configured `source_path`. Set `DATA.inputs.video.modality: skeleton_cache`
for cached experiments; keep `representation: skeleton_graph`. Train and evaluate
with the existing `train()`/`evaluate()` APIs. Shuffle cached windows, never live
association sessions. GPU extraction is not performed inside DataLoader workers.
No soccer coordinate normalization, team flips, graph mixup, or RGB augmentation
is applied. The initial spotting dataset enumerates shot-local overlapping windows
rather than soccer's foreground-resampled epoch schedule.

For direct-video inference using a known cache, set preprocessing `cache` to
`{mode: read, path: /absolute/cache/directory}`. The source video hash and exact
extraction settings must match. Cached manifest inputs skip frontend construction.
Do not change extraction sample rate without rebuilding caches. Use the same
class order and sample rate for training and inference.

Classification evaluation uses OSL's existing classification metrics. Skeleton
spotting evaluation uses existing `compute_mAPs_E2E` with 0–4 second tolerances
and existing `ForegroundF1` on the strongest emitted event at each frame
(`event_foreground_f1`). That event metric differs from raw, pre-NMS frame F1;
training validation still reports the existing dense frame metric.

## Representation and identity

A canonical observation contains `[P,17,2]` float32 image-pixel joints,
`[P,17]` raw confidence and boolean joint validity, `[P,4]` xyxy boxes, player
validity, track IDs, shot ID, image dimensions, near/far roles, source frame/PTS/time
base, video-relative timestamp, and sampling-grid timestamp. Timestamps are
relative to the first decoded presentation timestamp; original PTS are retained.
The sampler emits at most one observation per decoded frame and skips unavailable
grid points; it never fabricates intermediate poses. Missing/non-monotonic PTS
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
extraction settings, and resolved model checksums/runtime versions. Action-model
weights and augmentations are excluded. New extraction settings produce new caches.
The cache stores observations, not augmented tensors or graph objects.

Named model presets contain versioned upstream artifact URLs. The resolved ONNX
SHA-256 is recorded at initialization. Pin deployments using:

```yaml
params:
  preprocessing:
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

For action evaluation, pre-extract once, train the existing action models, and
report classification accuracy/balanced accuracy/macro-F1 and spotting mAP/F1
per class. Include bounce classes where present and report them separately;
body-only inputs may not identify ball bounces well. Compare cache/live inputs
and predictions before measuring accuracy. Run short and repeated long sources to
check that streaming RSS/VRAM plateau; complete result objects grow with event count.

Required contribution check: `bash scripts/run_tests.sh`. Fast tests use synthetic
backend outputs and real action models. Actual GPU/provider/weight checks belong
to opt-in release verification. Numerical performance and domain quality targets
remain unset until this public pilot has run. ShotQuality validation follows it.
