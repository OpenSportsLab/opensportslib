"""Basic VQA evaluation metrics for OpenSportsLib."""

from __future__ import annotations

import re
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


def _canonical_token_set(text: str) -> set[str]:
    expanded = (
        _normalize(text)
        .replace("no card", "yellow")
        .replace("red card", "red")
        .replace("yellow card", "yellow")
        .replace("without contact", "without_contact")
        .replace("with contact", "with_contact")
    )
    cleaned = re.sub(r"[^a-z0-9_ ]+", " ", expanded)
    tokens = [t for t in cleaned.split() if t]
    stop = {"the", "a", "an", "is", "it", "to", "of", "and", "because", "why"}
    return {t for t in tokens if t not in stop}


def _semantic_overlap(pred: str, refs: list[str]) -> float:
    pred_set = _canonical_token_set(pred)
    if not pred_set:
        return 0.0
    best = 0.0
    for ref in refs:
        ref_set = _canonical_token_set(ref)
        if not ref_set:
            continue
        inter = len(pred_set.intersection(ref_set))
        union = len(pred_set.union(ref_set))
        if union == 0:
            continue
        score = inter / union
        if score > best:
            best = score
    return best


def _semantic_card_match(pred: str, refs: list[str]) -> float:
    candidates = ("red", "yellow", "no card", "no offence", "foul")
    pred_norm = _normalize(pred)
    pred_hits = {c for c in candidates if c in pred_norm}
    if not pred_hits:
        return 0.0
    for ref in refs:
        ref_norm = _normalize(ref)
        if any(c in ref_norm for c in pred_hits):
            return 1.0
    return 0.0


def _semantic_foul_consistency(pred: str, refs: list[str]) -> float:
    pred_norm = _normalize(pred)
    pred_neg = any(t in pred_norm for t in ("no foul", "no offence", "fair challenge"))
    pred_pos = (any(t in pred_norm for t in ("yellow", "red", "penalty")) or ("foul" in pred_norm and not pred_neg))
    if not pred_pos and not pred_neg:
        return 0.0

    for ref in refs:
        ref_norm = _normalize(ref)
        ref_neg = any(t in ref_norm for t in ("no foul", "no offence", "fair challenge"))
        ref_pos = (any(t in ref_norm for t in ("yellow", "red", "penalty")) or ("foul" in ref_norm and not ref_neg))
        if (pred_pos and ref_pos) or (pred_neg and ref_neg):
            return 1.0
    return 0.0


def _semantic_rationale_quality(pred: str, refs: list[str]) -> float:
    del refs
    pred_norm = _normalize(pred)
    connector_count = sum(1 for t in ("because", "due to", "as ", "since") if t in pred_norm)
    detail_count = sum(1 for t in ("contact", "reckless", "challenge", "body", "advantage") if t in pred_norm)
    has_min_length = len(pred_norm.split()) >= 6
    if not has_min_length:
        return 0.0
    raw = 0.4 * min(connector_count, 1) + 0.6 * min(detail_count / 2.0, 1.0)
    return float(min(max(raw, 0.0), 1.0))


def compute_vqa_metrics(predictions: dict[str, Any], dataset, eval_profile: dict[str, Any] | None = None) -> dict[str, Any]:
    """Compute strict and semantic VQA metrics against reference answer lists."""
    eval_profile = eval_profile or {}
    pred_items = predictions.get("data", []) if isinstance(predictions, dict) else []
    if not pred_items:
        return {
            "exact_match": 0.0,
            "contains_match": 0.0,
            "token_f1": 0.0,
            "count": 0,
            "strict": {"exact_match": 0.0, "contains_match": 0.0, "token_f1": 0.0},
            "semantic": {"overlap_score": 0.0, "card_match": 0.0, "foul_consistency": 0.0, "rationale_quality": 0.0},
            "eval_profile": _normalize_eval_profile(eval_profile),
        }

    by_id_question = {}
    for sample in dataset:
        key = (sample.get("id"), sample.get("question"))
        by_id_question[key] = sample.get("references") or []

    exact = 0
    contains = 0
    total = 0
    token_f1_sum = 0.0
    semantic_overlap_sum = 0.0
    semantic_card_sum = 0.0
    semantic_foul_sum = 0.0
    semantic_rationale_sum = 0.0
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
        semantic_overlap_sum += _semantic_overlap(pred, refs)
        semantic_card_sum += _semantic_card_match(pred, refs)
        semantic_foul_sum += _semantic_foul_consistency(pred, refs)
        semantic_rationale_sum += _semantic_rationale_quality(pred, refs)

    if total == 0:
        return {
            "exact_match": 0.0,
            "contains_match": 0.0,
            "token_f1": 0.0,
            "count": 0,
            "strict": {"exact_match": 0.0, "contains_match": 0.0, "token_f1": 0.0},
            "semantic": {"overlap_score": 0.0, "card_match": 0.0, "foul_consistency": 0.0, "rationale_quality": 0.0},
            "eval_profile": _normalize_eval_profile(eval_profile),
        }
    metrics = {
        "exact_match": exact / total,
        "contains_match": contains / total,
        "token_f1": token_f1_sum / total,
        "count": total,
        "strict": {
            "exact_match": exact / total,
            "contains_match": contains / total,
            "token_f1": token_f1_sum / total,
        },
        "semantic": {
            "overlap_score": semantic_overlap_sum / total,
            "card_match": semantic_card_sum / total,
            "foul_consistency": semantic_foul_sum / total,
            "rationale_quality": semantic_rationale_sum / total,
        },
        "eval_profile": _normalize_eval_profile(eval_profile),
    }
    return metrics


def _normalize_eval_profile(eval_profile: dict[str, Any]) -> dict[str, Any]:
    metric_set = eval_profile.get("metric_set", ["exact_match", "contains_match", "token_f1", "semantic"])
    aggregation = str(eval_profile.get("aggregation", "mean"))
    exclusions = eval_profile.get("exclusions", [])
    return {
        "metric_set": list(metric_set) if isinstance(metric_set, (list, tuple)) else [str(metric_set)],
        "aggregation": aggregation,
        "exclusions": list(exclusions) if isinstance(exclusions, (list, tuple)) else [],
    }
