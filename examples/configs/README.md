# Example Configs

This folder contains minimal configuration templates for OpenSportsLib.

These examples are meant to help users and contributors understand the expected YAML structure without overloading the top level README.

## Recommended contents

- `classification.yaml`
- `localization.yaml`
- additional task specific configs as the library grows

## Usage

Point the OpenSportsLib Python API to one of these configs.

Example:

```python
from opensportslib import model

my_model = model.classification(
    config="examples/configs/classification.yaml"
)
```

## Notes

These files should stay:

- minimal
- readable
- aligned with the current public API
- updated when config fields change
