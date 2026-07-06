"""Prediction-prior builders for VQA prompts."""

from __future__ import annotations

from typing import Any


_XVARS_ACTION_PRIORS = {
    0: "a tackle",
    1: "a foot duel",
    2: "a high leg",
    3: "holding",
    4: "pushing",
    5: "using his elbows or arms",
    6: "a shoulder challenge",
    7: "a simulation",
}

_XVARS_OFFENCE_PRIORS = {
    0: "and no foul",
    1: "foul and no card",
    2: "foul and a yellow card",
    3: "foul and a red card",
}


def build_generic_prediction_prior_text(pred: dict[str, Any] | None, *, fields: list[str] | None = None) -> str:
    """Build prediction prior text from explicitly configured prediction fields."""

    pred = pred or {}
    chunks: list[str] = []
    for field in [str(field).strip() for field in (fields or []) if str(field).strip()]:
        value = pred.get(field)
        if isinstance(value, dict):
            value = value.get("label", value.get("value"))
        if value is not None and str(value).strip():
            chunks.append(f"{field}={value}")
    return "; ".join(chunks)


def build_xvars_classifier_prior(action_index: int, offence_index: int) -> str:
    """Translate XVARS/XFoul classifier labels into backend-specific prompt priors."""

    action = _XVARS_ACTION_PRIORS.get(int(action_index), "")
    offence = _XVARS_OFFENCE_PRIORS.get(int(offence_index), "")
    if not action:
        return offence
    if offence.startswith("and "):
        return f"{action} {offence}"
    return f"{action}, {offence}" if offence else action


def build_xvars_referee_prior_from_prediction(pred: dict[str, Any] | None) -> str:
    """Map XVARS/XFoul referee predictions into prompt prior text."""

    pred = pred or {}
    action = str(pred.get("Action class") or pred.get("action") or "").strip()
    offence = str(pred.get("Offence") or pred.get("offence") or "").strip()
    severity = str(pred.get("Severity") or pred.get("severity") or "").strip()

    action_map = {
        "tackling": "a tackle",
        "standing tackling": "a foot duel",
        "elbowing": "using his elbows or arms",
        "holding": "holding",
        "high leg": "a high leg",
        "pushing": "pushing",
        "challenge": "a shoulder challenge",
        "dive": "a simulation",
    }
    action = action_map.get(action.lower(), action)

    if offence.lower() == "offence":
        offence = "foul"
    elif offence.lower() == "no offence":
        offence = "no foul"
    if severity == "3.0":
        severity = "yellow card"
    elif severity == "5.0":
        severity = "red card"
    elif severity == "1.0":
        severity = "no card"
    parts = [p for p in (action, offence, severity) if p]
    return ", ".join(parts)


def build_prediction_prior_text(
    pred: dict[str, Any] | None,
    *,
    adapter: str | None = None,
    fields: list[str] | None = None,
) -> str:
    """Build optional prediction-derived prior text using explicit config."""

    adapter_name = str(adapter or "").strip().lower()
    if not adapter_name:
        return build_generic_prediction_prior_text(pred, fields=fields)
    if adapter_name in {"generic", "fields"}:
        return build_generic_prediction_prior_text(pred, fields=fields)
    if adapter_name in {"xvars_referee", "xvars_xfoul"}:
        return build_xvars_referee_prior_from_prediction(pred)
    raise ValueError(
        f"Unsupported VQA prediction prior adapter '{adapter}'. "
        "Expected one of: generic, fields, xvars_referee, xvars_xfoul."
    )
