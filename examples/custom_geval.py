"""Create a custom professional-answer-quality G-Eval metric."""

from deepeval.test_case import LLMTestCase

from deepeval_llm_testing.azure_openai_client import create_azure_judge
from deepeval_llm_testing.dataset import load_evaluation_cases
from deepeval_llm_testing.evaluation.geval import build_professional_quality_metric
from deepeval_llm_testing.reporting import measure_and_capture, print_result


def main() -> None:
    case = next(item for item in load_evaluation_cases() if item.id == "professional-001")
    test_case = LLMTestCase(
        input=case.input,
        actual_output=case.actual_output,
        expected_output=case.expected_output,
        context=case.context,
    )
    metric = build_professional_quality_metric(create_azure_judge(), threshold=0.75)

    print(f"Input: {test_case.input}")
    print(f"Actual output: {test_case.actual_output}")
    print(f"Expected output: {test_case.expected_output}")
    print(f"Context: {test_case.context}")
    print_result(measure_and_capture(metric, test_case))


if __name__ == "__main__":
    main()
