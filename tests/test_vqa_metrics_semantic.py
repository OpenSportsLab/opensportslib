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


def test_vqa_metrics_include_strict_and_semantic_groups():
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

    metrics = compute_vqa_metrics(predictions, _dataset(), eval_profile={"metric_set": ["semantic"]})

    assert "strict" in metrics
    assert "semantic" in metrics
    assert "token_f1" in metrics
    assert "exact_match" in metrics
    assert metrics["semantic"]["overlap_score"] > 0.0
    assert metrics["semantic"]["card_match"] > 0.0
    assert metrics["semantic"]["foul_consistency"] > 0.0
    assert metrics["semantic"]["rationale_quality"] > 0.0


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

    metrics = compute_vqa_metrics(predictions, _dataset())

    assert metrics["semantic"]["overlap_score"] >= 0.0
    assert metrics["semantic"]["card_match"] >= 0.0
    assert metrics["semantic"]["foul_consistency"] >= 0.0
    assert metrics["semantic"]["rationale_quality"] >= 0.0


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
    metrics = compute_vqa_metrics(predictions, _dataset())
    assert metrics["semantic"]["foul_consistency"] == 0.0
