"""Answer-relevancy metric configuration."""

from deepeval.metrics import AnswerRelevancyMetric
from deepeval.models.base_model import DeepEvalBaseLLM


def build_answer_relevancy_metric(
    model: DeepEvalBaseLLM,
    *,
    threshold: float = 0.7,
) -> AnswerRelevancyMetric:
    """Measure how directly the output addresses the user's input."""

    return AnswerRelevancyMetric(
        threshold=threshold,
        model=model,
        include_reason=True,
        async_mode=False,
    )
