"""Factory for DeepEval's native Azure OpenAI judge model."""

from __future__ import annotations

from deepeval.models import AzureOpenAIModel

from deepeval_llm_testing.config import AzureOpenAISettings


def create_azure_judge(
    settings: AzureOpenAISettings | None = None,
) -> AzureOpenAIModel:
    """Create the model object passed to DeepEval metric constructors.

    DeepEval 4.1.7 supports ``AzureOpenAIModel`` directly. Passing the model
    explicitly keeps project behavior independent of machine-level DeepEval CLI
    configuration and makes test dependencies easy to audit.
    """

    resolved = settings or AzureOpenAISettings.from_environment()
    return AzureOpenAIModel(
        model=resolved.model_name,
        deployment_name=resolved.deployment_name,
        api_key=resolved.api_key,
        base_url=resolved.endpoint,
        api_version=resolved.api_version,
        temperature=0,
    )
