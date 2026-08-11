"""Consistent console-friendly metric result reporting for examples."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from deepeval.test_case import LLMTestCase


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    metric: str
    score: float
    threshold: float | None
    reason: str
    passed: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def measure_and_capture(metric: Any, test_case: LLMTestCase) -> EvaluationResult:
    """Measure once and expose the score, reason, threshold, and pass/fail state."""

    metric.measure(test_case)
    return EvaluationResult(
        metric=getattr(metric, "name", metric.__class__.__name__),
        score=float(metric.score),
        threshold=getattr(metric, "threshold", None),
        reason=getattr(metric, "reason", "") or "No reason returned.",
        passed=bool(metric.is_successful()),
    )


def print_result(result: EvaluationResult) -> None:
    """Print only evaluation data—never client settings or credentials."""

    print(f"Metric: {result.metric}")
    print(f"Score: {result.score:.3f}")
    print(f"Threshold: {result.threshold}")
    print(f"Reason: {result.reason}")
    print(f"Result: {'PASS' if result.passed else 'FAIL'}")
