"""A live application call followed by an independent DeepEval judge."""

from __future__ import annotations

import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase

from deepeval_llm_testing.evaluation.relevancy import build_answer_relevancy_metric
from deepeval_llm_testing.llm_service import AzureLLMService


@pytest.mark.azure
def test_live_azure_openai_answer_is_relevant(azure_settings, azure_judge) -> None:
    user_input = "In two sentences, explain why LLM outputs need specialized evaluation."
    response = AzureLLMService(azure_settings).generate(
        user_input,
        system_prompt="Answer accurately, directly, and professionally.",
    )
    test_case = LLMTestCase(input=user_input, actual_output=response)
    metric = build_answer_relevancy_metric(azure_judge, threshold=0.7)

    assert_test(test_case, [metric], run_async=False)
