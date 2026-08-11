"""Run one answer-relevancy evaluation and print the complete result."""

from deepeval.test_case import LLMTestCase

from deepeval_llm_testing.azure_openai_client import create_azure_judge
from deepeval_llm_testing.dataset import load_evaluation_cases
from deepeval_llm_testing.evaluation.relevancy import build_answer_relevancy_metric
from deepeval_llm_testing.reporting import measure_and_capture, print_result


def main() -> None:
    case = next(item for item in load_evaluation_cases() if item.id == "relevant-correct-001")
    test_case = LLMTestCase(input=case.input, actual_output=case.actual_output)
    metric = build_answer_relevancy_metric(create_azure_judge(), threshold=0.7)

    print(f"Input: {case.input}")
    print(f"Actual output: {case.actual_output}")
    print_result(measure_and_capture(metric, test_case))


if __name__ == "__main__":
    main()
