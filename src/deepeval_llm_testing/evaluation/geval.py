"""Custom G-Eval factories using DeepEval 4's SingleTurnParams API."""

from collections.abc import Sequence

from deepeval.metrics import GEval
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import SingleTurnParams


def build_custom_geval_metric(
    *,
    name: str,
    criteria: str,
    evaluation_params: Sequence[SingleTurnParams],
    model: DeepEvalBaseLLM,
    threshold: float = 0.75,
) -> GEval:
    """Build a use-case-specific LLM-as-a-judge metric."""

    return GEval(
        name=name,
        criteria=criteria,
        evaluation_params=list(evaluation_params),
        threshold=threshold,
        model=model,
        async_mode=False,
    )


def build_professional_quality_metric(
    model: DeepEvalBaseLLM,
    *,
    threshold: float = 0.75,
) -> GEval:
    """Evaluate correctness, completeness, professionalism, clarity, and helpfulness."""

    criteria = (
        "The response should directly answer the user's question, be factually correct "
        "based on the expected output and available context, cover the important details, "
        "use clear professional language, avoid unnecessary information, and provide "
        "helpful actionable guidance."
    )
    return build_custom_geval_metric(
        name="Professional Answer Quality",
        criteria=criteria,
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
            SingleTurnParams.CONTEXT,
        ],
        model=model,
        threshold=threshold,
    )
