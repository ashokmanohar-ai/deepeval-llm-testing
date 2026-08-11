from __future__ import annotations

import json

import pytest

from deepeval_llm_testing.dataset import load_evaluation_cases


@pytest.mark.deterministic
def test_sample_dataset_covers_required_scenarios() -> None:
    cases = load_evaluation_cases()
    categories = {case.category for case in cases}

    assert len(cases) >= 10
    assert {
        "correct_and_relevant",
        "irrelevant_answer",
        "partially_correct",
        "hallucinated_answer",
        "faithful_rag_answer",
        "unfaithful_rag_answer",
        "high_quality_professional",
        "poorly_written",
        "successful_agent_task",
        "failed_agent_task",
    } <= categories


@pytest.mark.deterministic
def test_duplicate_dataset_ids_are_rejected(tmp_path) -> None:
    payload = [
        {"id": "duplicate", "category": "one", "input": "a", "actual_output": "b"},
        {"id": "duplicate", "category": "two", "input": "c", "actual_output": "d"},
    ]
    path = tmp_path / "duplicate.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unique"):
        load_evaluation_cases(path)
