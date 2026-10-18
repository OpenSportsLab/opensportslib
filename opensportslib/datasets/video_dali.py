"""Optional bounded DALI request decoder for both existing video datasets.

Requests come from torch samplers, not a second reader shuffle/shard policy.
The initial adapter builds a one-request reader; startup is benchmarked as part
of the complete pipeline. No decoded video or unbounded pipeline pool is kept.
"""
from pathlib import Path
import tempfile

import numpy as np


def decode_selected(path, indices, clocks, info, device_id=0):
    import torch
    try:
        from nvidia.dali import fn, pipeline_def
    except ImportError as exc:
        raise ImportError('Raw DALI video needs nvidia-dali-cuda120>=2.2; use opensportslib setup --dali') from exc
    if not torch.cuda.is_available():
        raise RuntimeError('DALI video decoding requires an available CUDA device')
    if not hasattr(fn.experimental.readers, 'video'):
        raise RuntimeError('Install the timestamp-aware DALI >=2.2 profile')
    times = np.array([clocks[i][2] for i in indices])
    if len(indices) > 1 and not np.allclose(np.diff(times), np.diff(indices)/info['fps'], atol=1e-4, rtol=1e-4):
        raise ValueError('The DALI pose path requires CFR; use loader_backend=opencv for VFR')
    from .video_raw import uniform_indices
    if uniform_indices(indices[0], indices[-1]+1, len(indices)) != indices:
        raise ValueError('DALI clip requests must use uniform source frame sampling')
    with tempfile.TemporaryDirectory(prefix='osl-dali-') as directory:
        # Reader file lists are whitespace-delimited. A private symlink also
        # handles video paths containing spaces without copying video bytes.
        alias = Path(directory)/('video'+Path(path).suffix)
        alias.symlink_to(Path(path).resolve())
        listing = Path(directory)/'clips.txt'
        listing.write_text(f'{alias} 0 {indices[0]} {indices[-1]+1}\n')

        @pipeline_def
        def pipeline():
            return fn.experimental.readers.video(
                device='gpu', file_list=str(listing), file_list_format='frames',
                file_list_include_end=False, sequence_length=len(indices), uniform_sample=True,
                enable_frame_num='sequence', enable_timestamps=True,
                random_shuffle=False, prefetch_queue_depth=1)

        pipe = pipeline(batch_size=1, num_threads=1, device_id=int(device_id),
                        prefetch_queue_depth=1, exec_async=False, exec_pipelined=False)
        try:
            pipe.build()
            data, _, numbers, timestamps = pipe.run()
            actual = np.asarray(numbers.as_cpu()[0] if hasattr(numbers, 'as_cpu') else numbers[0]).reshape(-1)
            actual_times = np.asarray(timestamps.as_cpu()[0] if hasattr(timestamps, 'as_cpu') else timestamps[0]).reshape(-1)
            if actual.tolist() != indices:
                raise RuntimeError('DALI returned different source frames than requested')
            # Reader clocks may include the source origin; compare intervals.
            if len(times) > 1 and not np.allclose(actual_times-actual_times[0], times-times[0], atol=1e-4):
                raise RuntimeError('DALI source timestamps differ from source presentation clocks')
            tensor = torch.from_dlpack(data[0])
            return {index: tensor[offset].cpu().numpy().copy() for offset, index in enumerate(indices)}
        finally:
            del pipe


def iter_decoded(path, length, stride, info, first=0, stop=None):
    """One persistent reader per chronological video, with exact source IDs."""
    import torch
    import av
    from nvidia.dali import fn, pipeline_def
    if not torch.cuda.is_available():
        raise RuntimeError('DALI video decoding requires an available CUDA device')

    stop = info['count'] if stop is None else stop
    temporary = tempfile.TemporaryDirectory(prefix='osl-dali-stream-')
    alias = Path(temporary.name)/('video'+Path(path).suffix)
    alias.symlink_to(Path(path).resolve())
    listing = Path(temporary.name)/'interval.txt'
    listing.write_text(f'{alias} 0 {first} {stop}\n')

    @pipeline_def
    def pipeline():
        return fn.experimental.readers.video(device='gpu', file_list=str(listing),
            file_list_format='frames', file_list_include_end=False,
            sequence_length=length, stride=stride, step=length*stride, pad_mode='constant',
            enable_frame_num='sequence', enable_timestamps=True,
            random_shuffle=False, prefetch_queue_depth=1)

    pipe = pipeline(batch_size=1, num_threads=1, device_id=torch.cuda.current_device(),
                    prefetch_queue_depth=1, exec_async=False, exec_pipelined=False)
    first_time = None
    # DALI exposes floating timestamps, not original integer PTS/time bases.
    # A sequential clock pass preserves these exactly without a full index or
    # guessing source PTS from FPS. Include this CPU decode cost in benchmarks.
    container = av.open(str(path))
    stream = container.streams.video[0]
    stream.thread_count = 1
    source_clocks = enumerate(container.decode(stream))
    source_index, source_frame = -1, None
    clock_origin = info['origin']*info['time_base'][0]/info['time_base'][1]
    first_source_time = None
    try:
        pipe.build()
        for _ in range(int(np.ceil((stop-first)/(length*stride)))):
            data, _, numbers, timestamps = pipe.run()
            indices = np.asarray(numbers.as_cpu()[0] if hasattr(numbers,'as_cpu') else numbers[0]).reshape(-1)
            clocks = np.asarray(timestamps.as_cpu()[0] if hasattr(timestamps,'as_cpu') else timestamps[0]).reshape(-1)
            tensor = torch.from_dlpack(data[0])
            for offset, (index, clock) in enumerate(zip(indices, clocks)):
                if index < 0:
                    continue
                if first_time is None:
                    if index != first:
                        raise RuntimeError('DALI did not preserve the source interval start frame')
                    first_time = float(clock)
                while source_index < index:
                    source_index, source_frame = next(source_clocks)
                if source_frame.pts is None:
                    raise ValueError('Video frames require presentation timestamps')
                timestamp = float(source_frame.pts*source_frame.time_base)-clock_origin
                if first_source_time is None:
                    first_source_time = timestamp
                if not np.isclose(timestamp, index/info['fps'], atol=1e-4, rtol=1e-6):
                    raise ValueError('The DALI pose path requires CFR; use loader_backend=opencv for VFR')
                if not np.isclose(float(clock)-first_time, timestamp-first_source_time, atol=1e-4, rtol=1e-6):
                    raise RuntimeError('DALI source timestamps differ from presentation clocks')
                time_base = [source_frame.time_base.numerator, source_frame.time_base.denominator]
                yield int(index), tensor[offset].cpu().numpy().copy(), int(source_frame.pts), time_base, timestamp
    finally:
        container.close()
        del pipe
        temporary.cleanup()
