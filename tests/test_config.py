from __future__ import annotations

import pytest

from deepeval_llm_testing.config import AzureOpenAISettings, ConfigurationError


@pytest.mark.deterministic
def test_loads_valid_azure_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key-not-a-real-secret")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example-resource.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME", "judge-deployment")
    monkeypatch.setenv("AZURE_OPENAI_MODEL_NAME", "gpt-4.1")

    settings = AzureOpenAISettings.from_environment()

    assert settings.deployment_name == "judge-deployment"
    assert settings.model_name == "gpt-4.1"
    assert settings.temperature == 0


@pytest.mark.deterministic
def test_rejects_non_https_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-key-not-a-real-secret")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "http://insecure.example.test")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME", "judge-deployment")

    with pytest.raises(ConfigurationError, match="complete HTTPS URL"):
        AzureOpenAISettings.from_environment()


@pytest.mark.deterministic
def test_rejects_placeholder_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "your-api-key")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com/")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT_NAME", "judge-deployment")

    with pytest.raises(ConfigurationError, match="AZURE_OPENAI_API_KEY"):
        AzureOpenAISettings.from_environment()
