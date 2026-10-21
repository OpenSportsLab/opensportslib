import importlib.util
import json
from pathlib import Path


def test_soccer_adapter_uses_full_rate_fallback_and_retains_late_ground_truth(tmp_path):
    from opensportslib.datasets.tdeed import load_videos

    script = Path(__file__).parents[3] / "tools/reproduction/prepare_tdeed_soccer.py"
    spec = importlib.util.spec_from_file_location("prepare_soccer", script)
    preparation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preparation)
    source, fallback, reference, output, media = [
        tmp_path / x for x in ["source", "fallback", "reference", "output", "media"]
    ]
    for p in [source, fallback, reference]:
        p.mkdir()
    for split, native in [("train", "train"), ("valid", "val"), ("test", "test")]:
        base = f"league/{split}"
        events = [
            {"label": "Goal", "position_ms": "41"},
            {"label": "Goal", "position_ms": "441"},
        ]
        row = {
            "id": "old",
            "inputs": [{"type": "video", "path": base + "/1_224p.mkv", "fps": 17.2}],
            "events": events,
        }
        high = {
            "id": "old",
            "inputs": [{"type": "video", "path": base + "/1_720p.mkv", "fps": 25}],
            "events": events,
        }
        (source / (split + ".json")).write_text(json.dumps({"data": [row]}))
        (fallback / (split + ".json")).write_text(json.dumps({"data": [high]}))
        (reference / (native + ".json")).write_text(
            json.dumps([{"video": base + "/half1", "num_frames": 10}])
        )
    assets = preparation.normalize(
        source, reference, output, media, "soccernet", fallback
    )
    assert len(assets) == 3 and all("720p" in x["path"] for x in assets)
    videos = load_videos(output / "train.json")
    assert videos[0].identity == "league/train/half1"
    assert videos[0].fps == 25 and videos[0].frames == 10
    assert [e["frame"] for e in videos[0].events] == [1, 11]
    assert "source-720p" in str(videos[0].path)
