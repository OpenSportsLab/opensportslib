"""Explicit diagnostic overlays with frame/track/confidence alignment."""
from .cache import SkeletonCache
from .session import decode_video

COCO_EDGES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11),
              (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]


def _draw_trace_pose(image, observation):
    import cv2
    import numpy as np
    for slot, present in enumerate(observation['player_valid']):
        if not present:
            continue
        color = (80, 255, 80) if slot % 2 == 0 else (0, 220, 255)
        points = np.rint(observation['joints'][slot]).astype(int)
        valid = observation['joint_valid'][slot]
        for a, b in COCO_EDGES:
            if valid[a] and valid[b]:
                cv2.line(image, tuple(points[a]), tuple(points[b]), color, 2)
        for point, visible in zip(points, valid):
            if visible:
                cv2.circle(image, tuple(point), 2, color, -1)
        x1, y1, x2, y2 = np.rint(observation['boxes'][slot]).astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
        role = {-1: 'unknown', 0: 'far', 1: 'near'}[observation['roles'][slot]]
        cv2.putText(image, f"ID {observation['track_ids'][slot]} {role}",
                    (x1, max(12, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, .35, color, 1)


def _draw_caption(image, lines):
    import cv2
    scale = max(.25, min(.65, image.shape[1] / 850))
    spacing = max(13, int(30 * scale))
    height = min(image.shape[0], spacing * len(lines) + 5)
    image[:height] = (image[:height].astype('float32') * .3).astype('uint8')
    for index, line in enumerate(lines):
        cv2.putText(image, line, (5, spacing * (index + 1)),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)


def render_prediction_overlay(video_path, trace_path, predictions, output_path, *,
                              event_threshold=.5, event_window_seconds=.25):
    """Render actual model observations at exact source frames, with predictions.

    All decoded frames and their PTS spacing are retained. No poses are inferred,
    held, or interpolated during rendering. Output is H.264 video without audio.
    """
    from collections import deque
    from fractions import Fraction
    from pathlib import Path
    import av
    import cv2
    import numpy as np
    from .diagnostics import trace_frames

    if not 0 <= event_threshold <= 1 or not np.isfinite(event_window_seconds) or event_window_seconds < 0:
        raise ValueError('Require confidence in [0,1] and a finite nonnegative event window')
    output_path = Path(output_path)
    if output_path.resolve() == Path(video_path).resolve():
        raise ValueError('Overlay output must differ from its source video')
    if output_path.exists():
        raise FileExistsError(output_path)
    records = predictions['data']
    if len(records) != 1:
        raise ValueError('Overlay requires predictions for exactly one video')
    classification = predictions['task'] == 'action_classification'
    label = next(iter(records[0]['labels'].values())) if classification else None
    events = [] if classification else sorted(
        (event for event in records[0]['events'] if float(event['confidence']) >= event_threshold),
        key=lambda event: float(event['position_ms']))
    active, next_event = deque(), 0
    observations = trace_frames(trace_path, video_path)
    observation = next(observations, None)
    temporary = output_path.with_name(output_path.stem + '.part.mp4')
    writer = None
    durations = {}

    def mux(packet):
        # Preserve source durations instead of the encoder's nominal-rate
        # duration, especially at the tail of a VFR video.
        duration = durations.pop(packet.pts * packet.time_base)
        if duration > 0:
            packet.duration = int(duration / packet.time_base)
        writer.mux(packet)

    count = sampled = 0
    try:
        with av.open(str(video_path)) as source:
            stream = source.streams.video[0]
            stream.thread_count = 1
            rate = stream.average_rate or stream.guessed_rate
            if not rate or rate <= 0:
                raise ValueError('Overlay requires a valid source video rate')
            origin = None
            for index, frame in enumerate(source.decode(stream)):
                if frame.pts is None or frame.time_base is None:
                    raise ValueError('Overlay requires source presentation timestamps')
                if origin is None:
                    origin = frame.pts * frame.time_base
                clock = frame.pts * frame.time_base - origin
                timestamp = float(clock)
                image = frame.to_ndarray(format='bgr24')
                if writer is None:
                    writer = av.open(str(temporary), 'w', format='mp4')
                    encoded = writer.add_stream('libx264', rate=rate)
                    # Right/bottom padding preserves original image coordinates.
                    encoded.width, encoded.height = frame.width + frame.width % 2, frame.height + frame.height % 2
                    encoded.pix_fmt = 'yuv420p'
                    encoded.time_base = encoded.codec_context.time_base = frame.time_base
                    encoded.codec_context.thread_count = 1
                    # Matching decode/presentation order makes each MP4 sample
                    # duration describe the same source frame on VFR videos.
                    encoded.options = {'crf': '18', 'preset': 'veryfast', 'bf': '0'}
                current = None
                if observation is not None and observation['frame_index'] < index:
                    raise ValueError('Pose trace skipped a source frame while rendering')
                if observation is not None and observation['frame_index'] == index:
                    if (observation['pts'] != frame.pts
                            or Fraction(*observation['time_base']) != frame.time_base
                            or abs(observation['timestamp'] - timestamp) > 1e-6
                            or observation['image_size'] != [frame.width, frame.height]):
                        raise ValueError('Pose trace/source coordinate or timestamp mismatch')
                    current = observation
                    _draw_trace_pose(image, current)
                    sampled += 1
                    observation = next(observations, None)
                lines = [f"{timestamp:.3f}s  frame={index}  " +
                         (f"sampled shot={current['shot_id']}" if current else 'not sampled')]
                if current is not None and not current['view_valid']:
                    lines.append('view masked')
                if classification:
                    lines.append(f"clip: {label['label']}  {float(label['confidence']):.3f}")
                else:
                    while next_event < len(events) and float(events[next_event]['position_ms']) / 1000 <= timestamp + event_window_seconds:
                        active.append(events[next_event])
                        next_event += 1
                    while active and float(active[0]['position_ms']) / 1000 < timestamp - event_window_seconds:
                        active.popleft()
                    lines.extend(f"{event['label']} {float(event['confidence']):.2f} "
                                 f"@{float(event['position_ms']) / 1000:.3f}s"
                                 for event in sorted(active, key=lambda event: -float(event['confidence']))[:3])
                    if not active:
                        lines.append('no event above display threshold')
                _draw_caption(image, lines)
                if image.shape[:2] != (encoded.height, encoded.width):
                    image = cv2.copyMakeBorder(image, 0, encoded.height - frame.height,
                                              0, encoded.width - frame.width, cv2.BORDER_CONSTANT)
                result = av.VideoFrame.from_ndarray(image, format='bgr24')
                result.time_base = frame.time_base
                result.pts = int(clock / frame.time_base)
                result.duration = frame.duration
                durations[clock] = frame.duration * frame.time_base
                for packet in encoded.encode(result):
                    mux(packet)
                count += 1
            if writer is None:
                raise ValueError('Source video has no frames')
            if observation is not None:
                raise ValueError('Pose trace extends beyond the source video')
            for packet in encoded.encode():
                mux(packet)
            encoded_size = [encoded.width, encoded.height]
            writer.close()
            writer = None
        temporary.rename(output_path)
        return dict(source_frames=count, sampled_source_frames=sampled, audio=False,
                    encoded_size=encoded_size)
    finally:
        observations.close()
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)


def render_overlay(video_path, cache_path, output_path):
    import cv2
    from pathlib import Path
    cache = SkeletonCache(cache_path)
    from .cache import file_sha256
    if cache.manifest['identity'].get('video_sha256') != file_sha256(video_path):
        raise ValueError('Overlay video does not match skeleton cache')
    if Path(output_path).resolve() in {Path(video_path).resolve(), Path(cache_path).resolve()}:
        raise ValueError('Overlay output must differ from its input')
    if Path(output_path).exists():
        raise FileExistsError(output_path)
    frames = iter(cache)
    observation = next(frames, None)
    writer = None
    try:
        for image, timestamp, index, _, _ in decode_video(video_path):
            if observation is None:
                break
            if index != observation.frame_index:
                continue
            h, w = image.shape[:2]
            if writer is None:
                fps = cache.manifest['identity']['settings'].get('sample_fps', 25)
                writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                if not writer.isOpened():
                    raise RuntimeError('Could not open overlay video encoder')
            for slot in range(len(observation.player_valid)):
                if not observation.player_valid[slot]:
                    continue
                color = (0, 220, 255) if slot % 2 else (80, 255, 80)
                points = observation.joints[slot].astype(int)
                valid = observation.joint_valid[slot]
                for a, b in COCO_EDGES:
                    if valid[a] and valid[b]:
                        cv2.line(image, tuple(points[a]), tuple(points[b]), color, 2)
                for joint in range(17):
                    if valid[joint]:
                        cv2.circle(image, tuple(points[joint]), 3, color, -1)
                x1, y1, x2, y2 = observation.boxes[slot].astype(int)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
                cv2.putText(image, f'ID {observation.track_ids[slot]} q={observation.confidence[slot].mean():.2f}',
                            (x1, max(15,y1)), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 1)
            cv2.putText(image, f'{timestamp:.3f}s shot={observation.shot_id} court={observation.view_valid}',
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1)
            writer.write(image)
            observation = next(frames, None)
    finally:
        if writer is not None:
            writer.release()
