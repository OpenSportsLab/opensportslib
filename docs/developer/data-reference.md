# Data Extension API

> **Extension point.** Each task owns its dataset selection. When adding a new
> input modality, keep the output format expected by that task's trainer.

## Routes and classes

| Module | Route/classes | Role |
| --- | --- | --- |
| `datasets.builder` | `build_dataset` | Routes `classification`, `localization`, and `vqa` tasks. |
| `datasets.classification_dataset` | `build`, `ClassificationDataset`, `VideoDataset`, `TrackingDataset`, `HFTrackingDataset`, `H5TrackingDataset` | Classification modalities. |
| `datasets.localization_dataset` | `LocalizationDataset` and concrete spotting/feature/SoccerNet classes | Localization dataset selection and sampling. |
| `datasets.vqa_dataset` | `VQADataset` | VQA manifest rows and answer references. |
| `datasets.utils.tracking` / `utils.h5_tracking` | tracking transforms, graph helpers, and HDF5 reader/manifest helpers | Tracking and HDF5 data support. |
| `datasets.hf_json`, `datasets.hf_tracking` | prepared-split types and `prepare_*_split` | Optional Hub-backed split staging. |

Dataset classes are **internal** implementations; the dataset-selection path is
the supported extension point. The following sections list non-private local
dataset classes, followed by the optional Hugging Face integration.

::: opensportslib.datasets.builder
    options:
      members:
        - build_dataset
      show_source: true

## Classification datasets

`classification_dataset.build(...)` chooses the configured classification
modality. `VideoDataset` handles video/frames-style inputs; `TrackingDataset`
handles tracking data; `HFTrackingDataset` and `H5TrackingDataset` are concrete
tracking specializations. These classes are consumed by
`Trainer_Classification`, so a new dataset must preserve the batch fields that
its selected trainer reads.

::: opensportslib.datasets.classification_dataset
    options:
      members:
        - build
        - ClassificationDataset
        - VideoDataset
        - TrackingDataset
        - HFTrackingDataset
        - H5TrackingDataset
      show_source: true

## Localization datasets

`LocalizationDataset` is the top-level localization entry point. It selects
video, tracking, feature, and SoccerNet-oriented concrete datasets based on
the configured split type and input modality. `FrameReader` and
`DatasetVideoSharedMethods` are shared implementation support for video routes.

| Dataset group | Concrete classes |
| --- | --- |
| Spotting/video | `ActionSpotDataset`, `ActionSpotVideoDataset` |
| Tracking spotting | `TrackingActionSpotDataset`, `TrackingActionSpotVideoDataset` |
| Feature JSON | `FeaturefromJson`, `FeatureClipsfromJSON`, `FeatureClipChunksfromJson` |
| SoccerNet | `SoccerNetGame`, `SoccerNetGameClips`, `SoccerNetGameClipsChunks`, `SoccerNet`, `SoccerNetClips`, `SoccerNetClipsChunks` |

::: opensportslib.datasets.localization_dataset
    options:
      members:
        - LocalizationDataset
        - FrameReader
        - ActionSpotDataset
        - ActionSpotVideoDataset
        - TrackingActionSpotDataset
        - TrackingActionSpotVideoDataset
        - FeaturefromJson
        - FeatureClipsfromJSON
        - FeatureClipChunksfromJson
        - SoccerNetGame
        - SoccerNetGameClips
        - SoccerNetGameClipsChunks
        - SoccerNet
        - SoccerNetClips
        - SoccerNetClipsChunks
      show_source: true

## VQA datasets

`VQADataset` converts an OSL JSON VQA split into rows used by the configured
VQA trainer and evaluator. It requires question/reference-answer data in the
manifest; model-specific feature generation belongs to the VQA model/trainer,
not this dataset route.

::: opensportslib.datasets.vqa_dataset
    options:
      members:
        - VQADataset
      show_source: true

## Tracking and HDF5 utilities

Tracking utilities normalize positions, build graph edge indices, cache games,
and apply spatial transforms. HDF5 utilities read timestamped tracking rows,
normalize their features, discover games, and write manifests. `write_h5_manifest`
writes a file; readers and cache builders perform local filesystem I/O.

::: opensportslib.datasets.utils.tracking
    options:
      members:
        - parse_frame
        - compute_deltas
        - normalize_features
        - build_edge_index
        - HorizontalFlip
        - VerticalFlip
        - TeamFlip
        - build_game_cache
        - load_or_build_game_cache
        - positions_to_strings
        - slice_window
      show_source: true

::: opensportslib.datasets.utils.h5_tracking
    options:
      members:
        - parse_utc
        - H5Frame
        - H5TrackingReader
        - normalize_h5_features
        - find_h5_games
        - write_h5_manifest
      show_source: true

## Hugging Face-backed inputs

Hub preparation is optional. It can download or materialize local files and
therefore requires a configured source, network access, and authentication for
gated repositories.

::: opensportslib.datasets.hf_json
    options:
      members:
        - PreparedHFJsonSplit
        - hf_json_source
        - prepare_hf_json_split
      show_source: true

::: opensportslib.datasets.hf_tracking
    options:
      members:
        - PreparedTrackingSplit
        - hf_tracking_source
        - prepare_hf_tracking_split
        - HFTarTrackingReader
      show_source: true
