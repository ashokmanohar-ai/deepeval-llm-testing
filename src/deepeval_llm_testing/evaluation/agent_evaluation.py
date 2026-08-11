"""Metrics for agent tool selection, observable outcomes, and full traces."""

from deepeval.metrics import GEval, TaskCompletionMetric, ToolCorrectnessMetric
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import SingleTurnParams, ToolCallParams

from deepeval_llm_testing.evaluation.geval import build_custom_geval_metric


class _DeterministicOnlyModel(DeepEvalBaseLLM):
    """Satisfy metric initialization without allowing an accidental LLM call."""

    def load_model(self):
        return None

    def generate(self, prompt: str, *args, **kwargs) -> str:
        raise RuntimeError("The deterministic tool metric unexpectedly requested an LLM judge.")

    async def a_generate(self, prompt: str, *args, **kwargs) -> str:
        raise RuntimeError("The deterministic tool metric unexpectedly requested an LLM judge.")

    def get_model_name(self) -> str:
        return "deterministic-tool-comparison"


def build_tool_correctness_metric(*, threshold: float = 1.0) -> ToolCorrectnessMetric:
    """Deterministically require expected tool names and arguments to match."""

    return ToolCorrectnessMetric(
        threshold=threshold,
        model=_DeterministicOnlyModel(),
        evaluation_params=[ToolCallParams.INPUT_PARAMETERS],
        should_exact_match=True,
        include_reason=True,
    )


def build_agent_outcome_metric(
    model: DeepEvalBaseLLM,
    *,
    threshold: float = 0.75,
) -> GEval:
    """Judge the observable outcome, not private chain-of-thought."""

    return build_custom_geval_metric(
        name="Agent Outcome Quality",
        criteria=(
            "Determine whether the agent completed the user's request, used the supplied "
            "tool result correctly, produced a relevant final answer, and avoided claims "
            "that are unsupported by the context. Evaluate only observable inputs, tool "
            "results, and outputs—not hidden chain-of-thought."
        ),
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT,
            SingleTurnParams.CONTEXT,
            SingleTurnParams.TOOLS_CALLED,
            SingleTurnParams.EXPECTED_TOOLS,
        ],
        model=model,
        threshold=threshold,
    )


def build_task_completion_metric(
    model: DeepEvalBaseLLM,
    *,
    task: str | None = None,
    threshold: float = 0.75,
) -> TaskCompletionMetric:
    """Build DeepEval's trace-based task-completion metric."""

    return TaskCompletionMetric(
        threshold=threshold,
        task=task,
        model=model,
        include_reason=True,
        async_mode=False,
    )
