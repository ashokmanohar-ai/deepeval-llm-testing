from __future__ import annotations

import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase

from deepeval_llm_testing.evaluation.hallucination import build_hallucination_metric


@pytest.mark.azure
def test_grounded_answer_has_low_hallucination(evaluation_cases, azure_judge) -> None:
    case = evaluation_cases["relevant-correct-001"]
    test_case = LLMTestCase(
        input=case.input,
        actual_output=case.actual_output,
        context=case.context,
    )
    metric = build_hallucination_metric(azure_judge, threshold=0.2)

    # Hallucination is lower-is-better: a score at or below 0.2 passes.
    assert_test(test_case, [metric], run_async=False)
