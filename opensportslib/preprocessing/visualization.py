"""Explicit diagnostic overlays with frame/track/confidence alignment."""
from .cache import SkeletonCache
from .session import decode_video

COCO_EDGES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11),
              (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]


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
