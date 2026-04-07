import pytest
import os
import json
import numpy as np
import yaml

@pytest.fixture(scope="session")
def mock_data_dir(tmp_path_factory):
    """Creates a temporary directory with mock datasets and configs."""
    base_dir = tmp_path_factory.mktemp("mock_data")
    
    # 1. Create mock npy files
    npy_path_1 = base_dir / "item1.npy"
    npy_path_2 = base_dir / "item2.npy"
    # shape: (T, H, W, C) for frames_npy
    dummy_video_1 = np.random.randint(0, 255, (4, 224, 224, 3), dtype=np.uint8)
    dummy_video_2 = np.random.randint(0, 255, (4, 224, 224, 3), dtype=np.uint8)
    np.save(npy_path_1, dummy_video_1)
    np.save(npy_path_2, dummy_video_2)

    # 2. Create Train Annotation JSON
    train_json = {
        "labels": {
            "action": {"labels": ["ClassA", "ClassB"]}
        },
        "data": [
            {
                "id": "1",
                "labels": {"action": {"label": "ClassA"}},
                "inputs": [{"type": "frames_npy", "path": "item1.npy"}]
            },
            {
                "id": "2",
                "labels": {"action": {"label": "ClassB"}},
                "inputs": [{"type": "frames_npy", "path": "item2.npy"}]
            }
        ]
    }
    
    train_file = base_dir / "train.json"
    with open(train_file, "w") as f:
        json.dump(train_json, f)

    test_file = base_dir / "test.json"
    with open(test_file, "w") as f:
        json.dump(train_json, f)

    # 3. Create mock yaml config
    config_dict = {
        "DATA": {
            "data_dir": str(base_dir),
            "data_modality": "frames_npy",
            "view_type": "single",
            "num_frames": 4,
            "input_fps": 1,
            "target_fps": 1,
            "start_frame": 0,
            "end_frame": 4,
            "normalize": True,
            "train_batch_size": 2,
            "test_batch_size": 2,
            "num_workers": 0
        },
        "MODEL": {
            "type": "custom",
            "pretrained_model": "r3d_18",
            "num_classes": 2,
            "edge": "spatial",
            "k": 5, "r": 2
        },
        "TRAIN": {
            "max_epochs": 1,
            "lr": 0.001
        }
    }
    
    config_file = base_dir / "classification.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_dict, f)

    return {
        "base_dir": str(base_dir),
        "train_json": str(train_file),
        "test_json": str(test_file),
        "config_yaml": str(config_file)
    }
