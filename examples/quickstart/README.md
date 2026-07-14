# Quickstart Examples

This folder contains small examples showing the intended public API of OpenSportsLib.

These scripts are meant to be:

- easy to read
- easy to adapt
- aligned with the README quickstart section

## Recommended examples

- `basic_classification.py`
- `basic_localization.py`
- `basic_vqa.py`

For VQA config templates, use `examples/configs/vqa_xvars.yaml` with
`opensportslib setup --vqa_xvars` or `examples/configs/vqa_qwen.yaml` with
`opensportslib setup --vqa_qwen`. The Qwen backend supports
`Qwen/Qwen2.5-7B-Instruct` and `Qwen/Qwen3.5-9B-Base`.

## Notes

Please keep these examples synchronized with the actual library API.
