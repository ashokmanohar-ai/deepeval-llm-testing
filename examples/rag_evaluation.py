"""Compare faithfulness, relevancy, and hallucination for one RAG answer."""

from deepeval.test_case import LLMTestCase

from deepeval_llm_testing.azure_openai_client import create_azure_judge
from deepeval_llm_testing.dataset import load_evaluation_cases
from deepeval_llm_testing.evaluation.faithfulness import build_faithfulness_metric
from deepeval_llm_testing.evaluation.hallucination import build_hallucination_metric
from deepeval_llm_testing.evaluation.relevancy import build_answer_relevancy_metric
from deepeval_llm_testing.reporting import measure_and_capture, print_result


def main() -> None:
    case = next(item for item in load_evaluation_cases() if item.id == "rag-faithful-001")
    judge = create_azure_judge()
    ground_truth = [
        "Unopened electronics can be returned within 30 days with a receipt or order number."
    ]
    test_case = LLMTestCase(
        input=case.input,
        actual_output=case.actual_output,
        expected_output=case.expected_output,
        context=ground_truth,
        retrieval_context=case.retrieval_context,
    )
    metrics = [
        build_answer_relevancy_metric(judge, threshold=0.7),
        build_faithfulness_metric(judge, threshold=0.8),
        build_hallucination_metric(judge, threshold=0.2),
    ]

    print(f"Question: {test_case.input}")
    print(f"Retrieval context: {test_case.retrieval_context}")
    print(f"Generated answer: {test_case.actual_output}")
    for metric in metrics:
        print("\n---")
        print_result(measure_and_capture(metric, test_case))


if __name__ == "__main__":
    main()
