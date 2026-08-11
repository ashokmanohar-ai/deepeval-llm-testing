"""Hallucination metric configuration for curated ground-truth context."""

from deepeval.metrics import HallucinationMetric
from deepeval.models.base_model import DeepEvalBaseLLM


def build_hallucination_metric(
    model: DeepEvalBaseLLM,
    *,
    threshold: float = 0.2,
) -> HallucinationMetric:
    """Detect contradictions against context; lower scores are better."""

    return HallucinationMetric(
        threshold=threshold,
        model=model,
        include_reason=True,
        async_mode=False,
    )
