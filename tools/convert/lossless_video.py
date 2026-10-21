"""Write RGB frames to FFV1 and verify every decoded pixel before publishing."""
from fractions import Fraction
import hashlib
from pathlib import Path


def write_lossless_frames(frames, output, fps):
    import av

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix('.part.mkv')
    rate = Fraction(fps).limit_denominator(1_000_000)
    expected, dimensions, count = hashlib.sha256(), None, 0
    try:
        with av.open(str(partial), 'w', format='matroska') as container:
            stream = container.add_stream('ffv1', rate=rate)
            stream.pix_fmt = 'bgr0'
            stream.codec_context.thread_count = 1
            for index, pixels in enumerate(frames):
                height, width = pixels.shape[:2]
                if dimensions is None:
                    dimensions = width, height
                    stream.width, stream.height = dimensions
                if dimensions != (width, height):
                    raise ValueError('Changing source dimensions')
                expected.update(pixels.tobytes())
                frame = av.VideoFrame.from_ndarray(pixels, format='rgb24')
                frame.pts, frame.time_base = index, 1/rate
                for packet in stream.encode(frame):
                    container.mux(packet)
                count += 1
            if not count:
                raise ValueError('Cannot encode an empty sequence')
            for packet in stream.encode():
                container.mux(packet)
        actual, decoded = hashlib.sha256(), 0
        with av.open(str(partial)) as container:
            if container.streams.video[0].average_rate != rate:
                raise ValueError('Output frame rate mismatch')
            container.streams.video[0].codec_context.thread_count = 1
            for frame in container.decode(video=0):
                actual.update(frame.to_ndarray(format='rgb24').tobytes())
                decoded += 1
        if decoded != count or actual.digest() != expected.digest():
            raise ValueError('Encoded video differs from source RGB frames')
        partial.replace(output)
        with output.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        return {'sha256':digest, 'rgb_sha256':expected.hexdigest(),
                'num_frames':count, 'width':dimensions[0], 'height':dimensions[1]}
    finally:
        partial.unlink(missing_ok=True)
