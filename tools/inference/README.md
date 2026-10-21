# Inspect a skeleton action model on one video

After training finishes, `inspect_video.py` loads a local classification or
localization checkpoint, runs inference, exports the **observations actually used
by that model**, and renders an annotated MP4. It uses the existing task wrappers,
classification video loader/trainer and spotting window/event logic.

Run from the repository root in the OSL environment, with the optional PyG and
GPU pose dependencies already installed:

```bash
conda activate osl

# Classification: supply one stroke clip. Choose your final best checkpoint.
python tools/inference/inspect_video.py \
  --task classification \
  --video /path/to/stroke.mp4 \
  --weights /path/to/classification/best/best_epoch_N.pt \
  --output-dir outputs/inspect-classification

# Localization: supply a continuous video or a shorter segment.
python tools/inference/inspect_video.py \
  --task localization \
  --video /path/to/match.mp4 \
  --weights /path/to/localization/best_checkpoint.pt \
  --output-dir outputs/inspect-localization \
  --event-threshold 0.5 --event-window-seconds 0.25
```

For both models on the same video, run the two commands with the same `--video`
and different weights/output directories. Their pose inputs can differ because
classification uniformly samples a clip and resets tracking per view, while
localization extracts chronologically and resets at cuts.

The script finds `config.yaml` beside the checkpoint, then one directory above.
Use `--config /path/to/saved/config.yaml` if needed. Use the configuration saved
with the checkpoint so graph depth, frame sampling, class order and frontend
settings match training. Configs with ordinary RGB encoders or precomputed soccer
tracking are rejected; this diagnostic requires `MODEL.components.pose_tracking`.

`--device cuda --gpu-id 0` overrides the saved device. Live pose extraction requires
CUDA; CPU mode is useful only with suitable existing classification caches.
Inference uses one process/GPU and zero video-loader workers. The script does
not launch training or access the training/test annotation files when their
class vocabulary is already present in the saved config.

## Outputs

The output directory must be new or empty. The source video and checkpoint are
never overwritten.

| File | Contents |
|---|---|
| `predictions.json` | Existing OSL clip-classification or spotting event JSON, saved explicitly through `save_predictions(...)`. |
| `poses.jsonl` | One JSON record per frontend observation consumed by the action route, including repeated classification samples. |
| `annotated.mp4` | H.264 video with selected player boxes, valid joints/bones, track IDs and predictions. |
| `run.json` | Completion status, video/checkpoint SHA-256 hashes, effective config, frontend provenance, joint order, observation counts and rendering settings. |
| `runtime/` | Task-wrapper working directories. |

`poses.jsonl` records `source`, `request_index`, `sample_index`, the loader's
`input_frame` metadata, canonical `skeleton`, and `[P,73]` `player_features`.
The skeleton contains source-image pixel joints, original joint confidence,
joint/player masks, `xyxy` boxes, shot-local track IDs, timestamps, PTS/time base,
source frame index, image dimensions, sampling-grid time, selection status and
near/far roles. Invalid/absent observations remain in the export with masks;
they are not drawn as players. Rejected detection candidates are not part of the
graph input and are not included.

Capture runs synchronously inside the **same preprocessing pass** used for model
inference. Exact-cache hits are captured too. Rendering reads this export and
decodes the video again; it does not run a second pose extractor. Classification
repeated frames remain repeated in JSON, with their individual sampling-grid
times, while the renderer draws each distinct source frame once.

## Prediction and display semantics

- **Classification:** one label/confidence for the entire supplied clip, displayed
  throughout that clip. A full broadcast still produces one clip label; it is
  not implicitly converted into rolling classification or action spotting.
- **Localization:** timestamped events from the existing shot-local overlapping
  windows and event suppression. The head is bidirectional; this is delayed
  windowed inference. The renderer runs after inference completes and does not
  imply causal or real-time results.
- Localization events with confidence at least `--event-threshold` are displayed
  within `±event-window-seconds` of the event time. These options affect the
  overlay only; saved model events remain unchanged. Up to three active events
  are shown, ordered by confidence.
- Boxes/poses appear **only at exact sampled source frames**. Other frames say
  `not sampled`. There is no pose holding or interpolation across frames/cuts.
  The banner shows clip/event predictions; scores are not assigned to individual
  track IDs because the action heads do not output actor IDs.
- All source video frames and presentation-time spacing are preserved; output
  starts at time zero. Output is silent. Odd source dimensions are padded by one
  pixel on the right/bottom for H.264, leaving pose coordinates unchanged.

Caching defaults to off in this diagnostic, independently of training's cache
policy. `--cache-dir data/tennis-classification/pose-cache` explicitly enables
classification exact-clip cache reads/writes. It only hits when source video,
frame/timing requests, decoder profile and extraction settings match. Continuous
localization inference keeps shot-local state and does not substitute the
independently initialized training-clip cache.

## Programmatic observation

An already constructed frontend exposes a scoped, opt-in observer:

```python
from opensportslib.preprocessing.diagnostics import PoseTraceWriter

with PoseTraceWriter("poses.jsonl") as writer, model.preprocessor.observe(writer):
    predictions = ...  # run this loaded model's normal forward/inference route
```

The callback receives `(records, video_meta)` before graph conversion, for live
clip forwards, exact-cache hits and continuous sessions. Records are borrowed
and must not be mutated. Exiting the context removes the observer even on error.
Ordinary inference continues to return predictions in memory without writing a
trace or overlay.

Offline CPU integration tests load real small action-model checkpoints and
encoded videos, replacing only the external pose backend. They cover both task
routes, exported graph features, repeated/padded/unsampled frames, cuts, masks,
cache hits, nonzero source PTS/VFR rendering and interruption cleanup. Run the
repository entry point `bash scripts/run_tests.sh`; real GPU pose quality and
performance remain separate opt-in checks.
