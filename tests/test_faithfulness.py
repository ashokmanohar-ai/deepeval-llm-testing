from __future__ import annotations

import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase

from deepeval_llm_testing.evaluation.faithfulness import build_faithfulness_metric


@pytest.mark.azure
def test_faithful_rag_answer_passes(evaluation_cases, azure_judge) -> None:
    case = evaluation_cases["rag-faithful-001"]
    test_case = LLMTestCase(
        input=case.input,
        actual_output=case.actual_output,
        retrieval_context=case.retrieval_context,
    )
    metric = build_faithfulness_metric(azure_judge, threshold=0.8)

    assert_test(test_case, [metric], run_async=False)
