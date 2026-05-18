# Example Configs

This folder contains minimal configuration templates for OpenSportsLib.

These examples are meant to help users and contributors understand the expected YAML structure without overloading the top level README.

## Recommended contents

- `classification_video.yaml`
- `classification_tracking.yaml`
- `localization.yaml`
- additional task specific configs as the library grows

## Usage

Point the OpenSportsLib Python API to one of these configs.

Example:

```python
from opensportslib.apis import ClassificationModel

my_model = ClassificationModel(
    config="examples/configs/classification_video.yaml"
)
```

## Notes

These files should stay:

- minimal
- readable
- aligned with the current public API
- updated when config fields change
