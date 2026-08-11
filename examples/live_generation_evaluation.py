"""Generate with Azure OpenAI, then judge the response with DeepEval."""

from deepeval.test_case import LLMTestCase

from deepeval_llm_testing.azure_openai_client import create_azure_judge
from deepeval_llm_testing.evaluation.geval import build_professional_quality_metric
from deepeval_llm_testing.llm_service import AzureLLMService
from deepeval_llm_testing.reporting import measure_and_capture, print_result


def main() -> None:
    question = "How should a team choose an LLM evaluation threshold?"
    expected = (
        "Use representative labeled examples, compare scores with human judgments, choose "
        "a threshold that reflects the acceptable false-pass risk, and monitor it over time."
    )
    service = AzureLLMService()
    answer = service.generate(question, system_prompt="Give concise, actionable guidance.")
    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        expected_output=expected,
        context=[expected],
    )

    print(f"Input: {question}")
    print(f"Actual output: {answer}")
    print(f"Expected output: {expected}")
    metric = build_professional_quality_metric(create_azure_judge())
    print_result(measure_and_capture(metric, test_case))


if __name__ == "__main__":
    main()
