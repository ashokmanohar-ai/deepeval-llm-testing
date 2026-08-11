"""Shared fixtures keep Azure configuration out of individual tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deepeval_llm_testing.azure_openai_client import create_azure_judge  # noqa: E402
from deepeval_llm_testing.config import (  # noqa: E402
    AzureOpenAISettings,
    ConfigurationError,
)
from deepeval_llm_testing.dataset import EvaluationCase, load_evaluation_cases  # noqa: E402


@pytest.fixture(scope="session")
def evaluation_cases() -> dict[str, EvaluationCase]:
    return {case.id: case for case in load_evaluation_cases()}


@pytest.fixture(scope="session")
def azure_settings() -> AzureOpenAISettings:
    try:
        return AzureOpenAISettings.from_environment()
    except ConfigurationError as exc:
        pytest.skip(f"Azure evaluation skipped: {exc}")


@pytest.fixture(scope="session")
def azure_judge(azure_settings: AzureOpenAISettings):
    return create_azure_judge(azure_settings)
