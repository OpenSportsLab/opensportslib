"""Explicit GPU benchmark of decode -> pose/tracking -> graph -> spotting.

Reports measurements only; never infers real-time capability from model-zoo FPS.
"""
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import threading
from time import perf_counter

import numpy as np


class MemorySampler:
    """Poll process RSS and total device memory, including ONNX Runtime."""
    def __init__(self, device):
        self.device = device
        self.rss = 0
        self.gpu = 0
        self.error = None
        self.stop = threading.Event()

    def sample(self):
        while not self.stop.is_set():
            try:
                status = Path('/proc/self/status').read_text()
                self.rss = max(self.rss, int(next(l for l in status.splitlines() if l.startswith('VmRSS:')).split()[1])*1024)
                result = subprocess.run(['nvidia-smi','--id',str(self.device),'--query-gpu=memory.used',
                                         '--format=csv,noheader,nounits'], capture_output=True,text=True,timeout=5,check=True)
                self.gpu = max(self.gpu, int(result.stdout.strip())*1024*1024)
            except (OSError, ValueError, StopIteration, subprocess.SubprocessError) as exc:
                self.error = str(exc)
            self.stop.wait(.1)

    def __enter__(self):
        self.thread = threading.Thread(target=self.sample,daemon=True)
        self.thread.start()
        return self

    def __exit__(self,*args):
        self.stop.set();self.thread.join(timeout=6)


def benchmark(config, video_path, output_path, repeats=3):
    import torch
    from opensportslib.models.builder import build_model
    from opensportslib.core.config.accessors import get_data_sampling
    from opensportslib.preprocessing.config import preprocessing_config
    from .backend import RTMLibBackend
    from .session import iter_skeletons, decode_video
    from .cache import file_sha256
    from opensportslib.apis._skeleton import spotting_events
    from opensportslib.apis.localization import LocalizationModel
    if str(config.TASK).lower() != 'localization':
        raise ValueError('Full-pipeline benchmark requires a skeleton localization config')
    if repeats < 1:
        raise ValueError('repeats must be positive')
    if Path(output_path).exists():
        raise FileExistsError(output_path)
    settings = preprocessing_config(config)
    if settings.get('cache',{}).get('mode','off') != 'off':
        raise ValueError('Benchmark live extraction with cache.mode=off')
    started = perf_counter()
    backend = RTMLibBackend(settings)
    device = settings.get('device','cuda:0')
    model = build_model(config,device)
    # Optional action checkpoint uses OSL's established checkpoint loader.
    checkpoint = getattr(config.MODEL.load,'checkpoint_path',None)
    if checkpoint:
        from opensportslib.core.config.editable import Config
        from opensportslib.core.config.runtime_adapter import namespace_to_plain_dict
        api = LocalizationModel(config=Config(namespace_to_plain_dict(config)))
        api.load_weights(checkpoint)
        model = api.model
    startup = perf_counter()-started
    model_seconds = [0.]
    class TimedModel:
        def predict(self,*args,**kwargs):
            before=perf_counter()
            result=model.predict(*args,**kwargs)
            model_seconds[0]+=perf_counter()-before
            return result
    source_count=0;source_duration=0.;shape=None
    for image,timestamp,*_ in decode_video(video_path):
        source_count+=1;source_duration=timestamp;shape=image.shape[:2]
    runs=[]
    try:
        # Explicit warmup, then measured runs with fresh per-video state.
        for run in range(repeats+1):
            arrivals={};latest=[0.];delay=[];processing=[];randomizer=random.Random(42)
            observations=[0];quality={'player_observations':0,'valid_joints':0,'view_samples':0,'id_changes':0,'shots':0}
            previous_ids={};previous_shot=[None]
            def frames():
                for frame in iter_skeletons(video_path,settings,backend):
                    arrivals[frame.timestamp]=perf_counter();latest[0]=frame.timestamp
                    observations[0]+=1
                    quality['player_observations']+=int(frame.player_valid.sum())
                    quality['valid_joints']+=int(frame.joint_valid.sum())
                    quality['view_samples']+=int(frame.view_valid)
                    if previous_shot[0]!=frame.shot_id:
                        previous_ids.clear();quality['shots']+=1;previous_shot[0]=frame.shot_id
                    for slot in np.flatnonzero(frame.player_valid):
                        identity=int(frame.track_ids[slot])
                        if slot in previous_ids and previous_ids[slot]!=identity:
                            quality['id_changes']+=1
                        previous_ids[slot]=identity
                    yield frame
            finalized=[0]
            def observer(frame,scores):
                elapsed=perf_counter()-arrivals.pop(frame.timestamp)
                lookahead=latest[0]-frame.timestamp
                finalized[0]+=1
                if len(delay)<4096:
                    delay.append(lookahead);processing.append(elapsed)
                else:
                    index=randomizer.randrange(finalized[0])
                    if index<4096:delay[index]=lookahead;processing[index]=elapsed
            before=perf_counter();first_result=None;events=0
            timings_before=dict(backend.timings);model_seconds[0]=0.
            with MemorySampler(str(device).split(':')[-1]) as memory:
                for batch in spotting_events(TimedModel(),config,frames(),observer=observer):
                    if first_result is None:first_result=perf_counter()-before
                    events+=len(batch)
            elapsed=perf_counter()-before
            if run:
                runs.append({'seconds':elapsed,'source_frames':source_count,'sampled_frames':observations[0],
                    'source_fps_processed':source_count/elapsed,'sample_fps_processed':observations[0]/elapsed,
                    'first_event_batch_seconds':first_result,'events':events,
                    'window_and_nms_lookahead_seconds_p50_p95':np.percentile(delay,[50,95]).tolist() if delay else [],
                    'post_extraction_finalization_seconds_p50_p95':np.percentile(processing,[50,95]).tolist() if processing else [],
                    'process_rss_peak_bytes':memory.rss,'device_used_peak_bytes':memory.gpu,'memory_sampling_error':memory.error,
                    'stage_seconds':{**{k:backend.timings[k]-timings_before[k] for k in backend.timings},'graph_action':model_seconds[0]},
                    'quality_counters_not_ground_truth_metrics':quality})
    finally:
        backend.close()
    hardware=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv'],capture_output=True,text=True,check=True).stdout
    report={'video':str(video_path),'video_sha256':file_sha256(video_path),'shape':shape,'duration_seconds':source_duration,
            'hardware':hardware,'cpu':platform.processor(),'python':platform.python_version(),
            'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'torch':torch.__version__,
            'action_checkpoint':checkpoint, 'action_weights_trained':bool(checkpoint),
            'provenance':backend.provenance,'settings':settings,'startup_seconds':startup,'runs':runs,
            'notes':['Device memory includes other processes; use an isolated GPU.',
                     'Sampled peaks at 100ms intervals are lower bounds on instantaneous peaks.',
                     'Finalization measurements start after pose extraction; lookahead is reported separately.',
                     'Quality counters require manual annotations for ID-switch/selection-error metrics.']}
    Path(output_path).write_text(json.dumps(report,indent=2))
    return report
