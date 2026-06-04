# Legacy to Canonical Mapping

This page documents migration behavior and key mapping between legacy and canonical config shapes.

## 1) Canonical Contract

- Runtime consumes canonical config only.
- Legacy config is accepted only at ingestion and migrated once.
- Canonical payloads containing legacy aliases are rejected.

## 8) Validation and Rejection Rules

Canonical validation enforces:
- required sections exist,
- `MODEL.task` equals `TASK`,
- component graph validity,
- no legacy aliases in canonical payload.

Forbidden in canonical payload (examples):
- top-level `dali`
- `DATA.annotations.*`
- `DATA.<split>.path` / `video_path`
- `MODEL.backbone` / `neck` / `head` / `post_proc`
- `TRAIN.num_epochs` / `TRAIN.max_epochs`

## 9) Migration Mapping Quick Reference

| Legacy | Canonical |
|---|---|
| `dali` | `DATA.common.runtime.loader_backend` |
| `DATA.<split>.path` | `DATA.common.splits.<split>.annotation_path` |
| `DATA.<split>.video_path` | `DATA.common.splits.<split>.source_path` |
| `MODEL.backbone` | `MODEL.components.*(kind=encoder)` |
| `MODEL.neck` | `MODEL.components.*(kind=adapter)` |
| `MODEL.head` | `MODEL.components.*(kind=head)` |
| `MODEL.post_proc` | `MODEL.components.*(kind=postprocessor)` |
| `TRAIN.num_epochs` / `TRAIN.max_epochs` | `TRAIN.epochs` |

