import importlib.util
import json
from pathlib import Path

import pytest


def campaign_module():
    script = Path(__file__).parents[3] / "tools/reproduction/tdeed.py"
    spec = importlib.util.spec_from_file_location("tdeed_campaign", script)
    campaign = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(campaign)
    return campaign


def test_main_campaign_rechecks_soccer_before_next_ready_run():
    campaign = campaign_module()
    entries = [
        {"id": "skating", "group": "released-main", "seeds": [1, 2]},
        {"id": "soccer", "group": "released-main", "seeds": [1, 2]},
        {"id": "bas_only", "group": "bas-ablation", "seeds": [1, 2]},
        {"id": "ablation", "group": "controlled-ablation", "seeds": [1, 2]},
    ]
    selected = campaign.select_stage(entries, "main", ["soccer"])
    assert [r["id"] for r in selected] == ["soccer", "skating"]
    ready = False

    def readiness(entry, seed):
        return {"ready": entry["id"] != "soccer" or ready, "errors": ["missing media"]}

    jobs = campaign.scheduled_runs(selected, readiness)
    blocked = [next(jobs), next(jobs)]
    assert all(e["id"] == "soccer" and check is not None for e, seed, check in blocked)
    entry, seed, check = next(jobs)
    assert (entry["id"], seed, check) == ("skating", 1, None)
    # Downloads finish while the skating run trains. Both soccer seeds must
    # precede the second skating seed without restarting the campaign.
    ready = True
    assert [(e["id"], seed, check) for e, seed, check in jobs] == [
        ("soccer", 1, None),
        ("soccer", 2, None),
        ("skating", 2, None),
    ]


def test_campaign_exits_when_only_blocked_jobs_remain():
    campaign = campaign_module()
    entries = [{"id": "soccer", "group": "released-main", "seeds": [1, 2]}]
    jobs = list(campaign.scheduled_runs(entries, lambda e, s: {"ready": False}))
    assert len(jobs) == 2
    assert all(check == {"ready": False} for e, s, check in jobs)


def test_existing_soccer_manifests_keep_pinned_asset_paths(tmp_path):
    script = Path(__file__).parents[3] / "tools/reproduction/prepare_tdeed_soccer.py"
    spec = importlib.util.spec_from_file_location("prepare_soccer", script)
    preparation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preparation)
    revisions = [
        "24fa021b2e74fe8ee76112a72ed942c61c268038",
        "8c98e425a8810602a806b4f17d42407354ebbd06",
    ]
    for split in ["train", "valid", "test"]:
        rows = [
            {
                "inputs": [{"path": f"{root}/{split}/half.mkv"}],
                "metadata": {"media_revision": rev},
            }
            for root, rev in zip(["source", "source-720p"], revisions)
        ]
        (tmp_path / (split + ".json")).write_text(json.dumps({"data": rows}))
    assets = preparation.existing_assets(tmp_path)
    assert len(assets) == 6
    assert sum(a["revision"] == revisions[1] for a in assets) == 3
    assert {a["root"] for a in assets} == {
        str(tmp_path / "source"), str(tmp_path / "source-720p")
    }
    rows[0]["inputs"][0]["path"] = "source/../../outside.mkv"
    (tmp_path / "train.json").write_text(json.dumps({"data": rows}))
    with pytest.raises(ValueError, match="Unexpected normalized media path"):
        preparation.existing_assets(tmp_path)

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
            "metadata": {"sha256": "old-encoding-checksum"},
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
    assert all(x["repo_id"] == preparation.OFFICIAL_REPO for x in assets)
    assert all(
        x["revision"] == preparation.OFFICIAL_REVISIONS["videos-720p"]
        for x in assets
    )
    videos = load_videos(output / "train.json")
    assert videos[0].identity == "league/train/half1"
    assert videos[0].fps == 25 and videos[0].frames == 10
    assert [e["frame"] for e in videos[0].events] == [1, 11]
    assert "source-720p" in str(videos[0].path)
    assert videos[0].sha256 is None
    metadata = json.loads((output / "train.json").read_text())["data"][0]["metadata"]
    assert metadata["annotation_source_media_sha256"] == "old-encoding-checksum"


def test_official_soccer_assets_keep_repo_branch_and_commit_together(tmp_path):
    script = Path(__file__).parents[3] / "tools/reproduction/prepare_tdeed_soccer.py"
    spec = importlib.util.spec_from_file_location("prepare_soccer", script)
    preparation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preparation)
    revision = preparation.OFFICIAL_REVISIONS["videos-224p"]
    for split in ["train", "valid", "test"]:
        row = {
            "inputs": [{"path": f"official/videos-224p/{split}/half.mkv"}],
            "metadata": {"media_repo_id": preparation.OFFICIAL_REPO,
                         "media_revision": revision},
        }
        (tmp_path / (split + ".json")).write_text(json.dumps({"data": [row]}))
    assets = preparation.existing_assets(tmp_path)
    assert len(assets) == 3
    assert all(x["repo_id"] == preparation.OFFICIAL_REPO for x in assets)
    assert all(x["revision"] == revision for x in assets)
    assert {x["path"] for x in assets} == {
        f"{split}/half.mkv" for split in ["train", "valid", "test"]
    }
    assert all(x["root"] == str(tmp_path / "official/videos-224p") for x in assets)
    # A real pinned commit from another branch must not be accepted as 224p.
    row["metadata"]["media_revision"] = preparation.OFFICIAL_REVISIONS["videos-720p"]
    (tmp_path / "test.json").write_text(json.dumps({"data": [row]}))
    with pytest.raises(ValueError, match="branch/revision mismatch"):
        preparation.existing_assets(tmp_path)


def test_soccer_audit_counts_frames_instead_of_trusting_metadata(tmp_path):
    import hashlib
    from tests.helpers.video_pose import write_video

    script = Path(__file__).parents[3] / "tools/reproduction/audit_soccernet_media.py"
    spec = importlib.util.spec_from_file_location("audit_soccer", script)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    # Eight encoded frames occupy a nominal ten-frame timeline: frame-count
    # estimates and end seeks cannot substitute for a sequential decode.
    path = write_video(tmp_path / "clip.mp4", count=8,
                       frame_indices=[0, 1, 2, 3, 4, 5, 8, 9])
    row = {
        "official_local_path": str(path),
        "official_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "expected_frames": 10,
    }
    result = audit.decode_file(row)
    assert result["decoded_frames"] == 8
    assert result["decoded_shortfall"] == 2
    assert result["pts_gaps_over_2ms"] > 0
    assert result["sha256_verified"] is True
    row["official_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256 differs"):
        audit.decode_file(row)
