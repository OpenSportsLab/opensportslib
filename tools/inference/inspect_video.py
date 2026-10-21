#!/usr/bin/env python3
"""Infer one video, export the actual model poses, and render predictions."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def inference_config(task, weights, output, config_path=None, device=None, gpu_id=None, cache_dir=None):
    from opensportslib.apis import Config
    from opensportslib.preprocessing.config import requires_raw_video
    from opensportslib.core.config.runtime_adapter import dict_to_namespace

    weights, output = Path(weights).resolve(), Path(output).resolve()
    if not weights.is_file():
        raise FileNotFoundError(f'Checkpoint not found: {weights}')
    if config_path is None:
        config_path = next((parent / 'config.yaml' for parent in (weights.parent, weights.parent.parent)
                            if (parent / 'config.yaml').is_file()), None)
    if config_path is None:
        raise ValueError('No saved config.yaml next to the checkpoint; supply --config explicitly')
    config = Config.from_file(config_path)
    document = config.get_config()
    if document['TASK'].lower() != task:
        raise ValueError(f"Checkpoint config is for {document['TASK']}, not {task}")
    if not requires_raw_video(dict_to_namespace(document)):
        raise ValueError('This script requires a model-owned pose_tracking video frontend')
    overrides = {'SYSTEM.paths.save_dir': str(output / 'runtime'),
                 'SYSTEM.paths.work_dir': str(output / 'runtime'),
                 'SYSTEM.paths.log_dir': str(output / 'runtime'),
                 'SYSTEM.gpu.count': 1, 'MODEL.runtime.multi_gpu': False}
    if 'multi_gpu' in document['TRAIN']['execution']:
        overrides['TRAIN.execution.multi_gpu'] = False
    frontend = next(name for name, component in document['MODEL']['components'].items()
                    if component['kind'] == 'preprocessor')
    overrides[f'MODEL.components.{frontend}.params.cache'] = (
        {'mode': 'read_write', 'path': str(Path(cache_dir).resolve())} if cache_dir else {'mode': 'off'})
    if device is not None:
        overrides['SYSTEM.device'] = device
    if gpu_id is not None:
        overrides['SYSTEM.gpu.id'] = gpu_id
    # This is one video; keep all pose sessions in the inference process.
    overrides.update({'DATA.common.splits.test.dataloader.batch_size': 1,
                      'DATA.common.splits.test.dataloader.num_workers': 0,
                      'DATA.common.splits.test.dataloader.pin_memory': False})
    config.update(overrides=overrides)
    return config, Path(config_path).resolve()


def run(task, video, weights, output, *, config_path=None, device=None, gpu_id=None,
        cache_dir=None, event_threshold=.5, event_window_seconds=.25):
    import math
    import torch
    from opensportslib.apis import ClassificationModel, LocalizationModel
    from opensportslib.preprocessing.cache import file_sha256
    from opensportslib.preprocessing.diagnostics import PoseTraceWriter
    from opensportslib.preprocessing.records import FEATURE_DIM, JOINT_NAMES
    from opensportslib.preprocessing.visualization import render_prediction_overlay

    video, weights, output = Path(video).resolve(), Path(weights).resolve(), Path(output).resolve()
    if task not in {'classification', 'localization'}:
        raise ValueError('Task must be classification or localization')
    if not video.is_file():
        raise FileNotFoundError(f'Video not found: {video}')
    if not 0 <= event_threshold <= 1 or not math.isfinite(event_window_seconds) or event_window_seconds < 0:
        raise ValueError('Require confidence in [0,1] and a finite nonnegative event window')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Choose a new or empty output directory: {output}')
    config, saved_config = inference_config(task, weights, output, config_path, device, gpu_id, cache_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = dict(status='running', task=task, source=str(video), source_sha256=file_sha256(video),
                  weights=str(weights), weights_sha256=file_sha256(weights), saved_config=str(saved_config),
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  joint_order=list(JOINT_NAMES), graph_feature_dim=FEATURE_DIM,
                  coordinates='source-image pixels, xyxy boxes',
                  pose_rendering='exact sampled frames only; no interpolation or holding',
                  prediction_semantics='one label for the entire clip' if task == 'classification'
                      else 'timestamped events from delayed bidirectional windows',
                  event_display_threshold=event_threshold, event_display_window_seconds=event_window_seconds)

    def save_report():
        (output / 'run.json').write_text(json.dumps(report, indent=2) + '\n')

    save_report()
    api, frontend = None, None
    try:
        wrapper = ClassificationModel if task == 'classification' else LocalizationModel
        api = wrapper(config=config, weights=str(weights))
        frontend = api.model.preprocessor
        with PoseTraceWriter(output / 'poses.jsonl') as trace, frontend.observe(trace), torch.no_grad():
            if task == 'classification':
                # Public classification infer builds a fresh worker model. Use
                # its existing task trainer with the loaded, observed instance.
                from opensportslib.core.utils.direct_video import direct_video_manifest
                from opensportslib.datasets.builder import build_dataset
                with direct_video_manifest(api.config, video, task) as manifest:
                    dataset = build_dataset(api.config, manifest, processor=None, split='test')
                    predictions = api.trainer.infer(dataset)
            else:
                predictions = api.infer(video_path=str(video), weights=str(weights), use_wandb=False)
        if not trace.observations:
            raise RuntimeError('The model produced no pose observations')
        api.save_predictions(str(output / 'predictions.json'), predictions)
        report.update(observation_requests=trace.requests, observations=trace.observations,
                      frontend_settings=frontend.settings,
                      frontend_provenance=[backend.provenance for backend in frontend._backends.values()],
                      effective_config=api.get_config())
        report['render'] = render_prediction_overlay(
            video, output / 'poses.jsonl', predictions, output / 'annotated.mp4',
            event_threshold=event_threshold, event_window_seconds=event_window_seconds)
        report['status'] = 'complete'
        save_report()
        return report
    except Exception as error:
        report.update(status='failed', error=str(error))
        save_report()
        raise
    finally:
        if frontend is not None:
            frontend.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', required=True, choices=('classification', 'localization'))
    parser.add_argument('--video', required=True, type=Path)
    parser.add_argument('--weights', required=True, type=Path, help='A completed local .pt checkpoint')
    parser.add_argument('--config', type=Path, help='Defaults to saved config.yaml beside the checkpoint or one directory above')
    parser.add_argument('--output-dir', required=True, type=Path, help='New or empty directory for all outputs')
    parser.add_argument('--device', choices=('cuda', 'cpu'), help='Defaults to the saved configuration')
    parser.add_argument('--gpu-id', type=int)
    parser.add_argument('--cache-dir', type=Path, help='Opt into exact-clip cache reuse/writes (classification)')
    parser.add_argument('--event-threshold', type=float, default=.5, help='Localization display threshold; saved predictions are unchanged')
    parser.add_argument('--event-window-seconds', type=float, default=.25, help='Show events this long before/after their timestamp')
    args = parser.parse_args(argv)
    try:
        result = run(args.task, args.video, args.weights, args.output_dir, config_path=args.config,
                     device=args.device, gpu_id=args.gpu_id, cache_dir=args.cache_dir,
                     event_threshold=args.event_threshold, event_window_seconds=args.event_window_seconds)
        print(f"Saved {result['observations']} model observations, predictions and annotated.mp4 to {args.output_dir}")
        return 0
    except Exception as error:
        print(f'Error: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
