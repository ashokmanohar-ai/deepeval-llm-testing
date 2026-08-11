from __future__ import annotations

import pytest
from deepeval import assert_test
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.test_case import LLMTestCase
from deepeval.tracing import observe

from deepeval_llm_testing.agent.mock_agent import knowledge_base_search, run_refund_agent
from deepeval_llm_testing.evaluation.agent_evaluation import (
    build_agent_outcome_metric,
    build_task_completion_metric,
    build_tool_correctness_metric,
)
from deepeval_llm_testing.evaluation.faithfulness import build_faithfulness_metric


@pytest.mark.deterministic
def test_agent_selected_correct_tool_and_arguments() -> None:
    run = run_refund_agent("Can I return an unopened headset delivered 10 days ago?")
    test_case = LLMTestCase(
        input=run.input,
        actual_output=run.actual_output,
        tools_called=run.tools_called,
        expected_tools=run.expected_tools,
    )
    metric = build_tool_correctness_metric()

    metric.measure(test_case)

    assert metric.score == 1
    assert metric.is_successful()


@pytest.mark.azure
def test_agent_final_response_is_grounded_and_complete(azure_judge) -> None:
    run = run_refund_agent("Can I return an unopened headset delivered 10 days ago?")
    test_case = LLMTestCase(
        input=run.input,
        actual_output=run.actual_output,
        expected_output=run.expected_output,
        context=run.context,
        retrieval_context=run.context,
        tools_called=run.tools_called,
        expected_tools=run.expected_tools,
    )
    metrics = [
        build_faithfulness_metric(azure_judge, threshold=0.8),
        build_agent_outcome_metric(azure_judge, threshold=0.75),
    ]

    assert_test(test_case, metrics, run_async=False)


@observe()
def traced_knowledge_search(query: str) -> str:
    return knowledge_base_search(query)


@observe()
def traced_refund_agent(query: str) -> str:
    policy = traced_knowledge_search(query)
    return f"The policy says: {policy}"


@pytest.mark.azure
def test_multi_step_agent_task_completion(azure_judge) -> None:
    metric = build_task_completion_metric(
        azure_judge,
        task="Find the internal refund policy and answer the user's return question.",
        threshold=0.75,
    )
    dataset = EvaluationDataset(
        goldens=[Golden(input="Can I return an unopened electronic item after 10 days?")]
    )

    # DeepEval attaches the metric to the full observed trajectory after the run.
    for golden in dataset.evals_iterator(metrics=[metric]):
        traced_refund_agent(golden.input)
