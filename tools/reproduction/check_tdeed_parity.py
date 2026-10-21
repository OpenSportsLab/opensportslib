"""Compare OSL with an independently supplied, pinned T-DEED source checkout."""

import argparse
import importlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    expected = "d953b4de8a2ad941c565e1683c5b07255cbe644e"
    revision = subprocess.check_output(
        ["git", "-C", str(a.reference_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != expected:
        raise ValueError("Parity requires the pinned upstream revision")
    sys.path.insert(0, str(a.reference_root))
    import torch
    import timm
    import numpy as np

    torch.set_num_threads(2)
    from opensportslib.models.base.tdeed import TDEEDModel, relocate_scores
    from opensportslib.models.utils.impl.tdeed.modules import EDSGPMIXERLayers

    reference = importlib.import_module("model.model")
    modules = importlib.import_module("model.modules")
    scorer = importlib.import_module("util.score")
    evaluator = importlib.import_module("util.eval")
    original = timm.create_model

    def offline(*args, **kwargs):
        kwargs["pretrained"] = False
        return original(*args, **kwargs)

    args = SimpleNamespace(
        modality="rgb",
        temporal_arch="ed_sgp_mixer",
        radi_displacement=2,
        feature_arch="rny002_gsf",
        clip_len=8,
        n_layers=2,
        sgp_ks=3,
        sgp_r=2,
        num_classes=4,
        crop_dim=None,
        pretrained=False,
    )
    with patch("timm.create_model", offline):
        torch.manual_seed(7)
        upstream = reference.TDEEDModel.Impl(args)
        torch.manual_seed(9)
        osl = TDEEDModel(args)
    osl.load(upstream.state_dict())
    upstream.eval()
    osl._model.eval()
    torch.manual_seed(11)
    rgb = torch.randint(0, 256, (2, 8, 3, 32, 32)).float()
    with torch.no_grad():
        ref, _ = upstream(rgb.clone(), inference=True)
        got, _ = osl._model(rgb.clone(), inference=True)
        for key in ref:
            torch.testing.assert_close(got[key], ref[key], rtol=0, atol=0)
        probabilities = modules.process_prediction(ref["im_feat"], ref["displ_feat"])
        actual = relocate_scores(got["im_feat"], got["displ_feat"])
        torch.testing.assert_close(actual, probabilities, rtol=0, atol=0)
        _, output = osl.predict(rgb, use_amp=False)
        np.testing.assert_array_equal(output, probabilities.numpy())
    upstream.train()
    osl._model.train()
    labels = torch.randint(0, 5, (2, 8))
    offsets = torch.randint(-2, 3, (2, 8)).float()
    torch.manual_seed(23)
    ref, _ = upstream(rgb.clone())
    loss = torch.nn.functional.cross_entropy(
        ref["im_feat"].reshape(-1, 5),
        labels.flatten(),
        weight=torch.tensor([1.0, 5, 5, 5, 5]),
    ) + torch.nn.functional.mse_loss(ref["displ_feat"], offsets)
    torch.manual_seed(23)
    got = osl.loss({"frame": rgb.clone(), "label": labels, "labelD": offsets})
    torch.testing.assert_close(loss, got, rtol=0, atol=0)
    loss.backward()
    got.backward()
    for (name, x), (other, y) in zip(
        upstream.named_parameters(), osl._model.named_parameters()
    ):
        assert name == other
        if x.grad is not None:
            torch.testing.assert_close(x.grad, y.grad, rtol=0, atol=0)
    from opensportslib.metrics.tdeed import frame_map, suppress

    truth = [
        {
            "video": "one",
            "events": [{"frame": 8, "label": "a"}, {"frame": 14, "label": "b"}],
        }
    ]
    events = [
        {"frame": i, "label": c, "score": float((i + 1) / 21)}
        for i in range(20)
        for c in ["a", "b"]
    ]
    predictions = [{"video": "one", "events": events}]
    maps, _ = scorer.compute_mAPs(truth, predictions, tolerances=[1, 2, 4])
    actual = frame_map(truth, predictions)
    assert maps == [actual["tolerances"][str(t)]["mAP"] for t in [1, 2, 4]]
    for soft in [False, True]:
        function = (
            evaluator.soft_non_maximum_supression
            if soft
            else evaluator.non_maximum_supression
        )
        result = function(predictions, 3, threshold=0.01)[0]["events"]
        assert result == suppress(events, 3, soft=soft)
    report = {
        "reference_commit": revision,
        "device": "cpu",
        "pretrained": False,
        "state_dict_compatible": True,
        "inference_logits_bit_exact": True,
        "relocation_bit_exact": True,
        "loss_bit_exact": True,
        "gradients_bit_exact": True,
        "PES_AP_exact": True,
        "hard_and_soft_NMS_exact": True,
        "scope": "200MF GSF, 8-frame/32px tensor fixture; benchmark results not evaluated",
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
