# Example Configs

This folder contains small OpenSportsLib configuration templates.

Use these examples to understand the expected YAML structure without adding all
of that detail to the top-level README.

## Recommended contents

- `classification_video.yaml`
- `classification_sngar_tracking.yaml`
- `localization_video_dali.yaml`
- `vqa_xvars.yaml`
- `vqa_qwen.yaml`
- additional task specific configs as the library grows

## Usage

Pass one of these configuration paths to the OpenSportsLib Python API.

Example:

```python
from opensportslib.apis import ClassificationModel

my_model = ClassificationModel(
    config="examples/configs/classification_video.yaml"
)
```

For VQA, use `examples/configs/vqa_xvars.yaml` for the X-VARS-compatible
backend or `examples/configs/vqa_qwen.yaml` for the Qwen-compatible backend.
Install matching dependencies with `opensportslib setup --vqa_xvars` or
`opensportslib setup --vqa_qwen`. The Qwen config supports
`Qwen/Qwen2.5-7B-Instruct` and `Qwen/Qwen3.5-9B-Base`.

## Notes

These files should stay:

- minimal
- readable
- aligned with the current public API
- updated when config fields change
