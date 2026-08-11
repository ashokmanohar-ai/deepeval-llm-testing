"""Faithfulness metric configuration for RAG generators."""

from deepeval.metrics import FaithfulnessMetric
from deepeval.models.base_model import DeepEvalBaseLLM


def build_faithfulness_metric(
    model: DeepEvalBaseLLM,
    *,
    threshold: float = 0.8,
) -> FaithfulnessMetric:
    """Measure factual alignment with retrieval_context."""

    return FaithfulnessMetric(
        threshold=threshold,
        model=model,
        include_reason=True,
        async_mode=False,
    )
