"""Load and validate the repository's JSON evaluation dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    id: str
    category: str
    input: str
    actual_output: str
    expected_output: str | None = None
    context: list[str] = field(default_factory=list)
    retrieval_context: list[str] = field(default_factory=list)
    expected_tool: str | None = None
    should_pass: bool = True
    notes: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EvaluationCase:
        required = ("id", "category", "input", "actual_output")
        missing = [name for name in required if not payload.get(name)]
        if missing:
            raise ValueError(f"Evaluation case is missing required fields: {', '.join(missing)}")
        return cls(**payload)


def default_dataset_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "evaluation_cases.json"


def load_evaluation_cases(path: Path | None = None) -> list[EvaluationCase]:
    dataset_path = path or default_dataset_path()
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Evaluation dataset root must be a JSON list.")

    cases = [EvaluationCase.from_dict(item) for item in payload]
    identifiers = [case.id for case in cases]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Evaluation case IDs must be unique.")
    return cases
