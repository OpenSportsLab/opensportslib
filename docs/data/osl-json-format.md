# OSL JSON Format

OSL JSON is the canonical annotation format used across OpenSportsLab tools.
OpenSportsLib uses it for dataset manifests, ground-truth annotations, and
prediction payloads returned by the high-level APIs.

An OSL JSON file is a single JSON object with project metadata, a shared label
schema, and a `data` array of samples. Each sample points to one or more input
files and can carry task-specific annotations.

The current OpenSportsLib implementation supports classification,
localization, and VQA workflows. The format also reserves payloads for
description and dense description tasks so datasets can stay compatible with
the broader OpenSportsLab ecosystem.

## Minimal Structure

The smallest useful file is a JSON object with a `data` list. For training and
evaluation, include a root `labels` schema and task-specific sample payloads.

```json
{
  "version": "2.0",
  "date": "2026-05-19",
  "dataset_name": "soccer-demo",
  "description": "Example OSL dataset.",
  "modalities": ["video"],
  "metadata": {
    "sport": "soccer",
    "split": "train"
  },
  "labels": {
    "action": {
      "type": "single_label",
      "labels": ["pass", "shot", "foul"]
    }
  },
  "data": []
}
```

## Top-Level Fields

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `version` | string | Recommended | Current canonical version is `"2.0"`. |
| `date` | string | Recommended | ISO date such as `"2026-05-19"`. |
| `dataset_name` | string | Recommended | Human-readable dataset or split name. |
| `description` | string | Optional | Free-text dataset description. |
| `modalities` | array[string] | Recommended | Input types present in `data[].inputs[]`, such as `["video"]`. |
| `metadata` | object | Optional | Dataset-level custom metadata. |
| `labels` | object | Required for supervised tasks | Shared label schema by annotation head. |
| `data` | array[object] | Required | Sample list. Must be a list. |

Unknown top-level keys are preserved by conversion tools where possible. Keep
custom dataset metadata under `metadata` unless another key is part of a
documented workflow.

## Label Schema

The root `labels` object defines annotation heads. Each head name is a key, and
each definition should include:

| Field | Type | Notes |
| --- | --- | --- |
| `type` | string | Use `single_label` for one class per sample/event, or `multi_label` for several labels. |
| `labels` | array[string] | Allowed class names for this head. |

```json
{
  "labels": {
    "action": {
      "type": "single_label",
      "labels": ["pass", "shot", "foul"]
    },
    "attributes": {
      "type": "multi_label",
      "labels": ["left_foot", "header", "set_piece"]
    }
  }
}
```

OpenSportsLib classification currently reads the `action` head by default:
`data[].labels.action.label`. Localization event heads should also point to the
same root schema, for example `data[].events[].head == "action"`.

## Sample Objects

Each entry in `data` is one sample.

| Field | Type | Notes |
| --- | --- | --- |
| `id` | string | Stable sample ID. Required for reliable evaluation and prediction matching. |
| `inputs` | array[object] | Media, feature, or tracking files for the sample. |
| `metadata` | object | Optional sample-level metadata such as match, game, clip, or timing fields. |
| `labels` | object | Classification annotations keyed by label head. |
| `events` | array[object] | Timestamped localization events. |
| `captions` | array[object] | Clip-level description captions. |
| `dense_captions` | array[object] | Timestamped dense descriptions. |
| `answers` | array[object] | Grouped question/answer annotations. |

Unknown sample keys are preserved by conversion tools where possible.

## Input Objects

Every sample should include `inputs`, even if it has only one input file.

```json
{
  "inputs": [
    {
      "type": "video",
      "path": "clips/clip_0001.mp4",
      "fps": 25.0
    }
  ]
}
```

Supported input types used by current OpenSportsLib workflows:

| Type | Typical path | Notes |
| --- | --- | --- |
| `video` | `clips/clip_0001.mp4` | Raw video clip or full game video. |
| `frames_npy` | `frames/clip_0001.npy` | NumPy frame array. The legacy alias `frame_npy` is normalized by annotation tooling. |
| `tracking_parquet` | `tracking/clip_0001.parquet` | Parquet tracking data. |
| `player_centroids_h5` | `tracking/live_centroids.h5` | Columnar H5 player centroid rows keyed by `timestamp_utc`, with player `x/y` coordinates. |
| `player_joints_h5` | `tracking/live_joints.h5` | Columnar H5 player joint rows keyed by `timestamp_utc`, with joint columns named like `nose_x`, `nose_y`, `nose_z`. |

`fps` is recommended for video and frame-array inputs. Tracking inputs can also
include `fps` as a fallback when timestamps are not available.

H5 player tracking inputs may include `ball_path` to point at a synchronized
ball H5 file. Ball files are resolved relative to the same split `source_path`
as `path`, unless an absolute path is provided. H5 loaders use `timestamp_utc`
as the source of truth and can clip each sample with optional
`data[].metadata.start_utc` and `data[].metadata.end_utc` values. When those
metadata fields are absent, the loader uses the available H5 timestamp range.

### Relative Path Resolution

OpenSportsLib stores input paths as relative paths inside JSON whenever
possible.

- Classification and localization training/inference resolve `inputs[].path`
  from the configured split media root, usually
  `DATA.common.splits.<split>.source_path`.
- Feature-based localization resolves `inputs[0].path` from the configured
  feature directory.
- Conversion tools resolve `inputs[].path` from the `media_root` argument passed
  to `convert_json_to_parquet(...)` or the CLI wrapper.
- Hugging Face upload/download tools treat `inputs[].path` as repository paths
  relative to the JSON file or selected split.

Example directory layout:

```text
dataset/
├── train/
│   ├── clips/clip_0001.mp4
│   └── annotations_train.json
├── valid/
│   ├── clips/clip_0101.mp4
│   └── annotations_valid.json
└── test/
    ├── clips/clip_0201.mp4
    └── annotations_test.json
```

For the train split, set `DATA.common.splits.train.source_path` to
`dataset/train` and store the sample path as `clips/clip_0001.mp4`.

### Multi-Input And Multi-View Samples

Use multiple `inputs` entries when a sample has more than one synchronized view
or modality.

```json
{
  "id": "play_0001",
  "inputs": [
    {
      "type": "video",
      "path": "wide/play_0001.mp4",
      "fps": 25.0
    },
    {
      "type": "video",
      "path": "close/play_0001.mp4",
      "fps": 25.0
    },
    {
      "type": "tracking_parquet",
      "path": "tracking/play_0001.parquet"
    },
    {
      "type": "player_centroids_h5",
      "path": "tracking/live_centroids.h5",
      "ball_path": "tracking/live_ball.h5"
    }
  ]
}
```

Classification multi-view loading groups samples by `id` when the config uses
`DATA.view_type: multi`. The current grouping helper also supports IDs with a
`_view` suffix, such as `play_0001_view1` and `play_0001_view2`.

## Classification Payload

Classification labels live under `data[].labels`. The key under `labels` should
match a root label head.

```json
{
  "id": "clip_0001",
  "inputs": [
    {
      "type": "video",
      "path": "clips/clip_0001.mp4",
      "fps": 25.0
    }
  ],
  "labels": {
    "action": {
      "label": "shot"
    },
    "attributes": {
      "labels": ["left_foot", "set_piece"]
    }
  }
}
```

For classification training, OpenSportsLib expects `labels.action.label` by
default. Samples without labels can be used for test/inference splits, but
training and validation need labels.

Smart predictions may include a confidence score. The annotation-tool convention
is `confidence_score`; current OpenSportsLib prediction exporters use
`confidence`.

```json
{
  "labels": {
    "action": {
      "label": "shot",
      "confidence_score": 0.91
    }
  }
}
```

## Localization Payload

Localization annotations live under `data[].events`. Each event is a point
timestamp in milliseconds.

```json
{
  "events": [
    {
      "head": "action",
      "label": "pass",
      "position_ms": 1240
    },
    {
      "head": "action",
      "label": "shot",
      "position_ms": 4320,
      "gameTime": "1 - 00:04",
      "confidence_score": 0.84
    }
  ]
}
```

OpenSportsLib localization prefers `position_ms` when present. If
`position_ms` is missing, feature-based JSON loaders fall back to `gameTime`.
For predictions and evaluation, current OpenSportsLib spotting outputs use
`confidence`.

### Open video intervals

A localization record may declare the playable or annotated portions of one
physical video in `data[].metadata.intervals`. Interval bounds are absolute
milliseconds on the physical video and follow half-open
`[start_time_ms, end_time_ms)` semantics:

```json
{
  "id": "FWC2022-M64-1",
  "inputs": [{"type": "video", "path": "224p/M64-1.mp4"}],
  "metadata": {
    "annotation_status": "verified",
    "intervals": [
      {"period": "1H", "start_time_ms": 5524000, "end_time_ms": 8672000},
      {"period": "2H", "start_time_ms": 9579000, "end_time_ms": 12800000}
    ]
  },
  "events": [
    {"head": "action", "label": "Header", "position_ms": 5668100}
  ]
}
```

The OpenCV localization loader exposes each interval as an ordered logical
video while reading frames from the original physical file. Event timestamps
are rebased to the interval start. Continuous inference state, including a
test-time adapter, remains alive as the loader advances between intervals.

`annotation_status` controls how the logical segments are used:

- `verified`: included in inference and evaluation;
- `unlabeled`: included in inference, but excluded from evaluation;
- `excluded`: omitted from both inference and evaluation.

Records without `metadata.intervals` retain full-video behavior. Bounded video
intervals currently require the OpenCV backend; selecting DALI raises an error
instead of silently reading the wrong physical frames. Events outside declared
intervals and overlapping or invalid intervals are rejected.

## Description, Dense Description, And Q/A Payloads

These payloads are part of the OSL JSON ecosystem. They are useful for datasets
that need to round-trip through OpenSportsLab annotation tools. Q/A payloads
are used by the OpenSportsLib VQA workflow.

Clip-level captions:

```json
{
  "captions": [
    {
      "lang": "en",
      "text": "A quick attack ends with a shot on goal."
    }
  ]
}
```

Timestamped dense captions:

```json
{
  "dense_captions": [
    {
      "position_ms": 1100,
      "lang": "en",
      "text": "The midfielder plays a forward pass."
    },
    {
      "position_ms": 3650,
      "lang": "en",
      "text": "The striker shoots from inside the area."
    }
  ]
}
```

Grouped question/answer annotations:

```json
{
  "id": "clip_0001",
  "inputs": [
    {
      "type": "video",
      "path": "clips/clip_0001.mp4",
      "fps": 25.0
    }
  ],
  "answers": [
    {
      "question": "What card would you give? Why?",
      "answers": ["No card, because this is a fair challenge."]
    }
  ]
}
```

## Complete Classification Example

```json
{
  "version": "2.0",
  "date": "2026-05-19",
  "task": "action_classification",
  "dataset_name": "soccer-classification-demo",
  "description": "Clip-level action labels.",
  "modalities": ["video"],
  "metadata": {
    "sport": "soccer",
    "split": "train"
  },
  "labels": {
    "action": {
      "type": "single_label",
      "labels": ["pass", "shot", "foul"]
    },
    "attributes": {
      "type": "multi_label",
      "labels": ["left_foot", "header", "set_piece"]
    }
  },
  "data": [
    {
      "id": "clip_0001",
      "inputs": [
        {
          "type": "video",
          "path": "clips/clip_0001.mp4",
          "fps": 25.0
        }
      ],
      "labels": {
        "action": {
          "label": "shot"
        },
        "attributes": {
          "labels": ["left_foot"]
        }
      },
      "metadata": {
        "match_id": "match_01"
      }
    }
  ]
}
```

## Complete Localization Example

```json
{
  "version": "2.0",
  "date": "2026-05-19",
  "task": "action_spotting",
  "dataset_name": "soccer-localization-demo",
  "description": "Timestamped action events.",
  "modalities": ["video"],
  "metadata": {
    "sport": "soccer",
    "split": "train"
  },
  "labels": {
    "action": {
      "type": "single_label",
      "labels": ["pass", "shot", "save"]
    }
  },
  "data": [
    {
      "id": "attack_0001",
      "inputs": [
        {
          "type": "video",
          "path": "clips/attack_0001.mp4",
          "fps": 25.0
        }
      ],
      "events": [
        {
          "head": "action",
          "label": "pass",
          "position_ms": 1100,
          "gameTime": "1 - 00:01"
        },
        {
          "head": "action",
          "label": "shot",
          "position_ms": 3650,
          "gameTime": "1 - 00:04"
        }
      ]
    }
  ]
}
```

## Multi-Modal Tracking Example

```json
{
  "version": "2.0",
  "date": "2026-05-19",
  "task": "action_classification",
  "dataset_name": "soccer-gar-multimodal-demo",
  "description": "Frames and tracking inputs for one action sample.",
  "modalities": ["frames_npy", "tracking_parquet"],
  "labels": {
    "action": {
      "type": "single_label",
      "labels": ["PASS", "SHOT"]
    }
  },
  "data": [
    {
      "id": "train_000001",
      "inputs": [
        {
          "type": "frames_npy",
          "path": "frames_npy/train/train_000001.npy",
          "fps": 2.0
        },
        {
          "type": "tracking_parquet",
          "path": "tracking_parquet/train/train_000001.parquet"
        }
      ],
      "labels": {
        "action": {
          "label": "PASS"
        }
      },
      "metadata": {
        "game_id": "game_001",
        "position_ms": 124000,
        "source_fps": 30.0,
        "effective_fps": 2.0,
        "window_size": 16,
        "frame_interval": 15
      }
    }
  ]
}
```

## H5 Tracking Example

```json
{
  "version": "2.0",
  "date": "2026-07-20",
  "task": "action_classification",
  "dataset_name": "soccer-h5-tracking-demo",
  "modalities": ["player_centroids_h5"],
  "labels": {
    "action": {
      "type": "single_label",
      "labels": ["header", "other"]
    }
  },
  "data": [
    {
      "id": "match_01_window_0001",
      "inputs": [
        {
          "type": "player_centroids_h5",
          "path": "live_centroids.h5",
          "ball_path": "live_ball.h5"
        }
      ],
      "metadata": {
        "start_utc": "2022-12-03 16:00:02.000000",
        "end_utc": "2022-12-03 16:00:04.000000"
      },
      "labels": {
        "action": {
          "label": "header"
        }
      }
    }
  ]
}
```

## Prediction Payloads

`infer()` returns predictions as an in-memory dictionary. It does not require
the caller to provide an output path. Use `save_predictions(...)` when you want
to write that dictionary to disk.

Classification prediction example:

```json
{
  "version": "2.0",
  "task": "action_classification",
  "date": "2026-05-19",
  "metadata": {
    "type": "predictions"
  },
  "data": [
    {
      "id": "clip_0001",
      "labels": {
        "action": {
          "label": "shot",
          "confidence": 0.91
        }
      }
    }
  ]
}
```

Localization prediction example:

```json
{
  "version": "2.0",
  "date": "2026-05-19",
  "task": "action_spotting",
  "metadata": {
    "type": "predictions"
  },
  "data": [
    {
      "inputs": [
        {
          "type": "video",
          "path": "clips/attack_0001.mp4",
          "fps": 2.0
        }
      ],
      "events": [
        {
          "head": "action",
          "label": "shot",
          "frame": 73,
          "position_ms": 36500,
          "gameTime": "1 - 00:36",
          "confidence": 0.84
        }
      ]
    }
  ]
}
```

## Validation Checklist

- `data` is a list.
- Every supervised file has a root `labels` schema.
- Classification samples use `labels.action.label` unless your code explicitly
  passes a different task head.
- Localization samples use `events[].position_ms` whenever possible.
- Localization interval bounds use absolute physical-video milliseconds and
  half-open `[start_time_ms, end_time_ms)` semantics.
- Labels in samples/events are present in the matching root label list.
- `inputs[].path` resolves from the expected split root or conversion
  `media_root`.
- Sample IDs are stable and unique, especially for evaluation.
