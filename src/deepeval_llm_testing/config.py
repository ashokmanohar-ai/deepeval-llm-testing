"""Secure, typed configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlparse

from dotenv import load_dotenv


class ConfigurationError(ValueError):
    """Raised when required Azure OpenAI configuration is missing or invalid."""


def _required_environment_value(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value or value.startswith("your-"):
        raise ConfigurationError(f"Set {name} to a real value in the environment or .env file.")
    return value


@dataclass(frozen=True, slots=True)
class AzureOpenAISettings:
    """Validated settings shared by generation and evaluation clients."""

    api_key: str
    endpoint: str
    api_version: str
    deployment_name: str
    model_name: str
    temperature: float = 0.0

    @classmethod
    def from_environment(cls) -> AzureOpenAISettings:
        """Load settings without overwriting values already supplied by the process."""

        load_dotenv(override=False)
        deployment_name = _required_environment_value("AZURE_OPENAI_DEPLOYMENT_NAME")
        settings = cls(
            api_key=_required_environment_value("AZURE_OPENAI_API_KEY"),
            endpoint=_required_environment_value("AZURE_OPENAI_ENDPOINT"),
            api_version=_required_environment_value("AZURE_OPENAI_API_VERSION"),
            deployment_name=deployment_name,
            model_name=os.getenv("AZURE_OPENAI_MODEL_NAME", deployment_name).strip()
            or deployment_name,
            temperature=float(os.getenv("AZURE_OPENAI_TEMPERATURE", "0")),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        parsed = urlparse(self.endpoint)
        if parsed.scheme != "https" or not parsed.netloc:
            raise ConfigurationError(
                "AZURE_OPENAI_ENDPOINT must be a complete HTTPS URL, for example "
                "https://my-resource.openai.azure.com/."
            )
        if not 0 <= self.temperature <= 2:
            raise ConfigurationError("AZURE_OPENAI_TEMPERATURE must be between 0 and 2.")
