from opensportslib.core.config.loader import resolve_config
from opensportslib.core.config.migrate import migrate_config


def _legacy_localization_cfg() -> dict:
    return {
        "TASK": "localization",
        "dali": True,
        "DATA": {
            "data_dir": "/home/vorajv/dataset/BAS/",
            "classes": ["PASS", "SHOT"],
            "train": {
                "type": "VideoGameWithDali",
                "video_path": "/home/vorajv/dataset/BAS/train",
                "path": "/home/vorajv/dataset/BAS/train/train.json",
            },
            "valid": {
                "type": "VideoGameWithDali",
                "video_path": "/home/vorajv/dataset/BAS/valid",
                "path": "/home/vorajv/dataset/BAS/valid/valid.json",
            },
            "valid_data_frames": {
                "type": "VideoGameWithDaliVideo",
                "video_path": "/home/vorajv/dataset/BAS/valid",
                "path": "/home/vorajv/dataset/BAS/valid/valid.json",
            },
        },
        "MODEL": {"type": "E2E", "backbone": {"type": "rny008_gsm"}, "head": {"type": "gru"}},
        "TRAIN": {"type": "trainer_e2e", "num_epochs": 1},
        "SYSTEM": {"device": "cpu", "GPU": 0, "gpu_id": 0, "save_dir": "./checkpoints"},
    }


def test_resolve_config_keeps_canonical_split_sources():
    cfg = resolve_config(migrate_config(_legacy_localization_cfg(), as_namespace=False), as_namespace=False)
    assert cfg["DATA"]["common"]["splits"]["valid"]["annotation_path"].endswith("valid.json")
    assert cfg["DATA"]["common"]["splits"]["valid_data_frames"]["annotation_path"].endswith("valid.json")
