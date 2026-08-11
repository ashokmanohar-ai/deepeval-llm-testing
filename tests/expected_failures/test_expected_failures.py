"""Intentional red tests, excluded from the normal suite and CI gate."""

from __future__ import annotations

import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase

from deepeval_llm_testing.agent.mock_agent import run_refund_agent
from deepeval_llm_testing.evaluation.agent_evaluation import build_tool_correctness_metric
from deepeval_llm_testing.evaluation.faithfulness import build_faithfulness_metric
from deepeval_llm_testing.evaluation.hallucination import build_hallucination_metric
from deepeval_llm_testing.evaluation.relevancy import build_answer_relevancy_metric


def test_wrong_agent_tool_fails_deterministically() -> None:
    run = run_refund_agent("Can I return this item?", use_wrong_tool=True)
    test_case = LLMTestCase(
        input=run.input,
        actual_output=run.actual_output,
        tools_called=run.tools_called,
        expected_tools=run.expected_tools,
    )
    metric = build_tool_correctness_metric()
    metric.measure(test_case)

    assert metric.is_successful(), "Intentional failure: the agent selected the wrong tool."


@pytest.mark.azure
def test_irrelevant_answer_fails(evaluation_cases, azure_judge) -> None:
    case = evaluation_cases["irrelevant-001"]
    test_case = LLMTestCase(input=case.input, actual_output=case.actual_output)
    assert_test(test_case, [build_answer_relevancy_metric(azure_judge)], run_async=False)


@pytest.mark.azure
def test_hallucinated_answer_fails(evaluation_cases, azure_judge) -> None:
    case = evaluation_cases["hallucinated-001"]
    test_case = LLMTestCase(
        input=case.input,
        actual_output=case.actual_output,
        context=case.context,
    )
    assert_test(test_case, [build_hallucination_metric(azure_judge)], run_async=False)


@pytest.mark.azure
def test_unfaithful_rag_answer_fails(evaluation_cases, azure_judge) -> None:
    case = evaluation_cases["rag-unfaithful-001"]
    test_case = LLMTestCase(
        input=case.input,
        actual_output=case.actual_output,
        retrieval_context=case.retrieval_context,
    )
    assert_test(test_case, [build_faithfulness_metric(azure_judge)], run_async=False)
