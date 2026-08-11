from __future__ import annotations

import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase

from deepeval_llm_testing.evaluation.geval import build_professional_quality_metric


@pytest.mark.azure
def test_professional_answer_quality_passes(evaluation_cases, azure_judge) -> None:
    case = evaluation_cases["professional-001"]
    test_case = LLMTestCase(
        input=case.input,
        actual_output=case.actual_output,
        expected_output=case.expected_output,
        context=case.context,
    )
    metric = build_professional_quality_metric(azure_judge, threshold=0.75)

    assert_test(test_case, [metric], run_async=False)
