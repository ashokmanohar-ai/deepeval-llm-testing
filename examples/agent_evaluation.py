"""Evaluate mock-agent tool use, final output, grounding, and full trajectory."""

from deepeval.dataset import EvaluationDataset, Golden
from deepeval.test_case import LLMTestCase
from deepeval.tracing import observe

from deepeval_llm_testing.agent.mock_agent import knowledge_base_search, run_refund_agent
from deepeval_llm_testing.azure_openai_client import create_azure_judge
from deepeval_llm_testing.evaluation.agent_evaluation import (
    build_agent_outcome_metric,
    build_task_completion_metric,
    build_tool_correctness_metric,
)
from deepeval_llm_testing.evaluation.faithfulness import build_faithfulness_metric
from deepeval_llm_testing.reporting import measure_and_capture, print_result


@observe()
def traced_knowledge_search(query: str) -> str:
    return knowledge_base_search(query)


@observe()
def traced_agent(query: str) -> str:
    policy = traced_knowledge_search(query)
    return f"Based on the internal policy: {policy}"


def main() -> None:
    query = "Can I return an unopened headset delivered 10 days ago?"
    run = run_refund_agent(query)
    test_case = LLMTestCase(
        input=run.input,
        actual_output=run.actual_output,
        expected_output=run.expected_output,
        context=run.context,
        retrieval_context=run.context,
        tools_called=run.tools_called,
        expected_tools=run.expected_tools,
    )

    print("Agent steps:")
    for index, step in enumerate(run.steps, start=1):
        print(f"{index}. {step}")

    print("\nDeterministic tool evaluation")
    print_result(measure_and_capture(build_tool_correctness_metric(), test_case))

    judge = create_azure_judge()
    for metric in [
        build_faithfulness_metric(judge),
        build_agent_outcome_metric(judge),
    ]:
        print("\n---")
        print_result(measure_and_capture(metric, test_case))

    print("\nTrace-based task completion")
    task_completion = build_task_completion_metric(
        judge,
        task="Find the internal refund policy and answer the return question.",
    )
    dataset = EvaluationDataset(goldens=[Golden(input=query)])
    for golden in dataset.evals_iterator(metrics=[task_completion]):
        traced_agent(golden.input)
    print("DeepEval evaluated the observed agent and tool spans above.")


if __name__ == "__main__":
    main()
