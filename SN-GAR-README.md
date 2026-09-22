Here’s a clean path to reproduce the **tracking model with the highest reported balanced accuracy**: GIN + MaxPool + positional edges. The paper reports **77.8% balanced accuracy and 57.0% macro F1**, averaged over five runs. Its baseline config uses seed 42, so the steps below produce one run, which may differ from that average. [Paper repository](https://github.com/drishyakarki/pixels_vs_positions)

### 1. Set up OpenSportsLib

Run these commands from the OpenSportsLib repository root. If you already have the `osl` environment set up, you can keep using it.

```bash
conda create -n osl-gar python=3.12 pip -y
conda activate osl-gar
python -m pip install -e '.[hf-tracking]'
opensportslib setup
python -m pip install torch-geometric

python -c "import torch, torch_geometric; print('CUDA available:', torch.cuda.is_available())"
```

The paper’s config requests CUDA. Its reported training cost is about **4 GPU hours**. [Paper repository](https://github.com/drishyakarki/pixels_vs_positions)

The GIN tracking baseline uses `torch-geometric` without its optional compiled extensions. This avoids the unavailable ARM64 wheels for `torch-scatter` and `torch-sparse`. [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)

### 2. Get access to the tracking data

Request access on the [current SoccerNet-GAR dataset page](https://huggingface.co/datasets/OpenSportsLab/SoccerNet-GAR), then authenticate:

```bash
hf auth login
```

Gated datasets require both approved access and authentication. [Hugging Face documentation](https://huggingface.co/docs/hub/datasets-gated)

The [sngar_tracking_hf.yaml](/home/giancos/git/opensportslib/sngar_tracking_hf.yaml) config reads the train, valid, and test splits from the `tracking` branch. It streams each split's small metadata tables with `datasets`, then downloads TAR shards when the weighted sampler first requests a clip inside them. Shards remain in `/home/giancos/OSLdata/sngar/hf_cache`; individual Parquet clips are read in memory and are never extracted as separate files.

The paper's replacement sampler requests 40,000 clips per epoch. It can reach most shards early in the first epoch, so initial batches may wait for shard downloads. Later epochs and runs reuse the cached TARs. The branch is pinned to a Hub commit for each run, and cached metadata is scoped to that commit.

If you prefer the existing fully extracted dataset workflow, use the local config and download the splits:

```bash
for split in train valid test; do
  python tools/download/download_osl_hf.py \
    --repo-id OpenSportsLab/SoccerNet-GAR \
    --revision tracking \
    --split "$split" \
    --output-dir ~/OSLdata/sngar
done
```

This produces `/home/giancos/OSLdata/sngar/tracking/train/train.json`, and equivalent paths for `valid` and `test`, with the referenced tracking clips alongside them. The [current dataset card](https://huggingface.co/datasets/OpenSportsLab/SoccerNet-GAR) recommends this branch. Its layout differs from the zip files described in the paper repository’s older download instructions.

### 3. Use the paper’s config

Both [sngar_tracking_hf.yaml](/home/giancos/git/opensportslib/sngar_tracking_hf.yaml) and [sngar_tracking_local.yaml](/home/giancos/git/opensportslib/sngar_tracking_local.yaml) use OpenSportsLib's canonical v2 layout. They share the [paper baseline's](https://github.com/drishyakarki/pixels_vs_positions/blob/main/main_tracking_gin_positional_maxpool.yaml) GIN model, 100 epochs, batch size 32, 4,000 replacement samples per class per epoch, Adam at `0.001`, and seed 42. Only the data source and split paths differ: the HF config reads cached TAR shards, while the local config reads extracted Parquet clips under `/home/giancos/OSLdata/sngar/tracking`.

No clip extraction or full split download is needed for streaming. To check the config:

```bash
python - <<'PY'
from opensportslib.apis import Config
config = Config.from_file("sngar_tracking_hf.yaml").get_config()
assert config["TRAIN"]["sampling"]["samples_per_class"] == 4000
print(config["DATA"]["inputs"]["tracking"]["source"])
PY
```

### 4. Train and evaluate

Save this as `run_sngar_tracking.py` in the OpenSportsLib root:

```python
from opensportslib.apis import ClassificationModel


def main():
    model = ClassificationModel(config="sngar_tracking_hf.yaml")

    checkpoint = model.train(use_ddp=False, use_wandb=False)
    print("Best checkpoint:", checkpoint)

    predictions = model.infer(use_wandb=False)
    metrics = model.evaluate(predictions=predictions, use_wandb=False)

    print(f"Balanced accuracy: {metrics['balanced_accuracy'] * 100:.2f}%")
    print(f"Macro F1: {metrics['f1'] * 100:.2f}%")


if __name__ == "__main__":
    main()
```

Run it in a terminal from the repository root:

```bash
conda activate osl-gar
python run_sngar_tracking.py
```

OpenSportsLib selects the best validation checkpoint during training, then `infer()` uses that checkpoint for the test set. The config saves checkpoints under `checkpoints_tracking/graph_conv/`.

The GIN baseline is the tracking result to use for **balanced accuracy**. If by “best” you mean **macro F1**, the paper reports a slightly higher F1 for its TCN temporal aggregation config (58.4%, with 75.5% balanced accuracy). [Paper results](https://github.com/drishyakarki/pixels_vs_positions#full-results)
