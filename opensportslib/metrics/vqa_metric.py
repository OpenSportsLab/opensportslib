"""Basic VQA evaluation metrics for OpenSportsLib."""

from __future__ import annotations

from typing import Any


def _normalize(text: str) -> str:
    return " ".join(str(text).strip().lower().split())


def _token_f1(pred: str, refs: list[str]) -> float:
    pred_tokens = _normalize(pred).split()
    if not pred_tokens:
        return 0.0
    best = 0.0
    for ref in refs:
        ref_tokens = _normalize(ref).split()
        if not ref_tokens:
            continue
        common = 0
        ref_counts = {}
        for t in ref_tokens:
            ref_counts[t] = ref_counts.get(t, 0) + 1
        for t in pred_tokens:
            if ref_counts.get(t, 0) > 0:
                common += 1
                ref_counts[t] -= 1
        if common == 0:
            continue
        precision = common / len(pred_tokens)
        recall = common / len(ref_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        if f1 > best:
            best = f1
    return best


def compute_vqa_metrics(predictions: dict[str, Any], dataset) -> dict[str, float]:
    """Compute exact and contains match against reference answer lists."""
    pred_items = predictions.get("data", []) if isinstance(predictions, dict) else []
    if not pred_items:
        return {"exact_match": 0.0, "contains_match": 0.0, "count": 0}

    by_id_question = {}
    for sample in dataset:
        key = (sample.get("id"), sample.get("question"))
        by_id_question[key] = sample.get("references") or []

    exact = 0
    contains = 0
    total = 0
    token_f1_sum = 0.0
    for item in pred_items:
        key = (item.get("id"), item.get("question"))
        refs = by_id_question.get(key, [])
        if not refs:
            continue
        total += 1
        pred = _normalize(item.get("answer_text", ""))
        ref_norm = [_normalize(r) for r in refs]
        if pred in ref_norm:
            exact += 1
        if any(pred and (pred in r or r in pred) for r in ref_norm):
            contains += 1
        token_f1_sum += _token_f1(pred, refs)

    if total == 0:
        return {"exact_match": 0.0, "contains_match": 0.0, "token_f1": 0.0, "count": 0}
    return {
        "exact_match": exact / total,
        "contains_match": contains / total,
        "token_f1": token_f1_sum / total,
        "count": total,
    }
