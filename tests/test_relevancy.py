from __future__ import annotations

import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase

from deepeval_llm_testing.evaluation.relevancy import build_answer_relevancy_metric


@pytest.mark.azure
def test_high_relevancy_example_passes(evaluation_cases, azure_judge) -> None:
    case = evaluation_cases["relevant-correct-001"]
    test_case = LLMTestCase(input=case.input, actual_output=case.actual_output)
    metric = build_answer_relevancy_metric(azure_judge, threshold=0.7)

    assert_test(test_case, [metric], run_async=False)
