import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch


def fixture_config(tmp_path):
    import av
    from opensportslib.core.config.editable import Config

    root = Path(__file__).parents[3]
    doc = Config.from_file(
        root / "opensportslib/configs/localization/tdeed/FineDiving_small.yaml"
    ).get_config()
    doc["SYSTEM"].update(device="cpu")
    doc["SYSTEM"].setdefault("reproducibility", {}).update(seed=7, use_seed=True)
    doc["SYSTEM"]["paths"]["save_dir"] = str(tmp_path / "artifacts")
    video = doc["DATA"]["inputs"]["video"]
    video["sampling"].update(clip_len=8, epoch_num_frames=16)
    video["transform"]["resize"] = {"height": 32, "width": 32}
    video["params"]["mixup"] = True
    encoder = doc["MODEL"]["components"]["video_encoder"]["params"]
    encoder.update(pretrained=False, crop_dim=None)
    doc["MODEL"]["components"]["task_head"]["params"].update(
        n_layers=1, sgp_ks=3, sgp_r=2, radi_displacement=1
    )
    doc["TRAIN"]["epochs"] = 2
    doc["TRAIN"]["scheduler"]["warm_up_epochs"] = 1
    doc["TRAIN"]["execution"]["start_valid_epoch"] = 0
    doc["TRAIN"]["selection"]["test_flip"] = False
    for split, spec in doc["DATA"]["common"]["splits"].items():
        media = tmp_path / f"{split}.mkv"
        with av.open(str(media), "w") as container:
            stream = container.add_stream("ffv1", rate=25)
            stream.width = 32
            stream.height = 32
            stream.pix_fmt = "bgr0"
            stream.codec_context.thread_count = 1
            for i in range(20):
                frame = av.VideoFrame.from_ndarray(
                    np.full((32, 32, 3), i * 10, np.uint8), format="rgb24"
                )
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        manifest = tmp_path / f"{split}.json"
        events = [
            {"label": c, "frame": i * 4 + 2, "position_ms": (i * 4 + 2) * 40}
            for i, c in enumerate(doc["DATA"]["common"]["classes"])
        ]
        manifest.write_text(
            json.dumps(
                {
                    "data": [
                        {
                            "id": split,
                            "inputs": [
                                {"type": "video", "path": media.name, "fps": 25}
                            ],
                            "metadata": {"video": split, "num_frames": 20},
                            "events": events,
                        }
                    ]
                }
            )
        )
        spec.update(
            annotation_path=str(manifest), source_path=str(tmp_path), overlap_len=4
        )
        spec["dataloader"].update(batch_size=2, num_workers=0)
    return doc


def test_tdeed_real_api_train_resume_infer_evaluate_without_prediction_writes(
    tmp_path, monkeypatch
):
    from opensportslib.apis import LocalizationModel
    from opensportslib.core.config.editable import Config
    from opensportslib.models.base.tdeed import TDEEDModel

    torch.set_num_threads(2)
    doc = fixture_config(tmp_path)
    reference = LocalizationModel(Config(doc))
    reference.save_dir = str(tmp_path / "continuous")
    reference.train(use_wandb=False)
    expected = torch.load(Path(reference.save_dir) / "last.pt", weights_only=False)
    original = TDEEDModel.epoch
    calls = []

    def interrupt(self, *args, **kwargs):
        if kwargs.get("optimizer") is not None:
            calls.append(True)
            if len(calls) == 2:
                raise RuntimeError("intentional epoch-boundary interruption")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(TDEEDModel, "epoch", interrupt)
    interrupted = LocalizationModel(Config(doc))
    interrupted.save_dir = str(tmp_path / "resumed")
    with pytest.raises(RuntimeError, match="intentional"):
        interrupted.train(use_wandb=False)
    monkeypatch.setattr(TDEEDModel, "epoch", original)
    resumed = LocalizationModel(Config(doc))
    resumed.save_dir = interrupted.save_dir
    checkpoint = resumed.train(
        weights=str(Path(resumed.save_dir) / "last.pt"), use_wandb=False
    )
    actual = torch.load(Path(resumed.save_dir) / "last.pt", weights_only=False)
    assert expected["history"] == actual["history"]
    for key in expected["model_state_dict"]:
        torch.testing.assert_close(
            expected["model_state_dict"][key],
            actual["model_state_dict"][key],
            rtol=0,
            atol=0,
        )
    before = set(Path(resumed.save_dir).rglob("*"))
    predictions = resumed.infer(weights=checkpoint, use_wandb=False)
    assert set(Path(resumed.save_dir).rglob("*")) == before
    assert predictions["task"] == "localization" and predictions["data"][0]["events"]
    direct = resumed.infer(
        video_path=str(tmp_path / "test.mkv"), weights=checkpoint, use_wandb=False
    )
    assert direct["data"][0]["events"] == predictions["data"][0]["events"]
    assert (
        resumed.evaluate(predictions=predictions, use_wandb=False)["metric"]
        == "PES interpolated event AP"
    )
    report = resumed.evaluate(weights=checkpoint, use_wandb=False)
    assert set(report) == {"none", "nms", "soft_nms"}
    resumed.save_predictions(str(tmp_path / "predictions.json"), predictions)
    assert (tmp_path / "predictions.json").is_file()
