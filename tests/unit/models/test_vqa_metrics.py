from opensportslib.metrics.vqa_metric import compute_vqa_metrics


def _dataset():
    return [
        {
            "id": "action_1",
            "question": "What card would you give? Why?",
            "references": ["Yellow card because of reckless challenge with contact."],
        },
        {
            "id": "action_2",
            "question": "Is it a foul or not? Why?",
            "references": ["No offence due to fair shoulder challenge without contact."],
        },
    ]


def test_vqa_metrics_include_strict_and_referee_semantic_groups_when_requested():
    predictions = {
        "task": "vqa",
        "data": [
            {
                "id": "action_1",
                "question": "What card would you give? Why?",
                "answer_text": "Yellow card for reckless challenge with contact.",
            },
            {
                "id": "action_2",
                "question": "Is it a foul or not? Why?",
                "answer_text": "No offence because challenge looked fair and without contact.",
            },
        ],
    }

    metrics = compute_vqa_metrics(predictions, _dataset(), eval_profile={"metric_set": ["referee_semantic"]})

    assert "strict" in metrics
    assert "referee_semantic" in metrics
    assert "token_f1" in metrics
    assert "exact_match" in metrics
    assert metrics["referee_semantic"]["overlap_score"] > 0.0
    assert metrics["referee_semantic"]["card_match"] > 0.0
    assert metrics["referee_semantic"]["foul_consistency"] > 0.0
    assert metrics["referee_semantic"]["rationale_quality"] > 0.0


def test_semantic_overlap_handles_partial_match():
    predictions = {
        "task": "vqa",
        "data": [
            {
                "id": "action_1",
                "question": "What card would you give? Why?",
                "answer_text": "Yellow card.",
            },
            {
                "id": "action_2",
                "question": "Is it a foul or not? Why?",
                "answer_text": "Unclear.",
            },
        ],
    }

    metrics = compute_vqa_metrics(predictions, _dataset(), eval_profile={"metric_set": ["referee_semantic"]})

    assert metrics["referee_semantic"]["overlap_score"] >= 0.0
    assert metrics["referee_semantic"]["card_match"] >= 0.0
    assert metrics["referee_semantic"]["foul_consistency"] >= 0.0
    assert metrics["referee_semantic"]["rationale_quality"] >= 0.0


def test_semantic_penalizes_contradictory_answer():
    predictions = {
        "task": "vqa",
        "data": [
            {
                "id": "action_1",
                "question": "What card would you give? Why?",
                "answer_text": "No foul and no card.",
            }
        ],
    }
    metrics = compute_vqa_metrics(predictions, _dataset(), eval_profile={"metric_set": ["referee_semantic"]})
    assert metrics["referee_semantic"]["foul_consistency"] == 0.0


def test_default_metrics_stay_generic_without_referee_semantics():
    predictions = {
        "task": "vqa",
        "data": [
            {
                "id": "action_1",
                "question": "What card would you give? Why?",
                "answer_text": "Yellow card.",
            }
        ],
    }

    metrics = compute_vqa_metrics(predictions, _dataset())

    assert "semantic" not in metrics
    assert "referee_semantic" not in metrics
    assert metrics["eval_profile"]["metric_set"] == ["exact_match", "contains_match", "token_f1"]


def test_generic_semantic_metric_name_does_not_enable_referee_scoring():
    predictions = {
        "task": "vqa",
        "data": [
            {
                "id": "action_1",
                "question": "What card would you give? Why?",
                "answer_text": "Yellow card.",
            }
        ],
    }

    metrics = compute_vqa_metrics(predictions, _dataset(), eval_profile={"metric_set": ["semantic"]})

    assert "semantic" not in metrics
    assert "referee_semantic" not in metrics
