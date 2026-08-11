"""Application-side Azure OpenAI text generation, separate from evaluation."""

from __future__ import annotations

from openai import AzureOpenAI

from deepeval_llm_testing.config import AzureOpenAISettings


class AzureLLMService:
    """Small generation client used by the live LLM unit-test example."""

    def __init__(self, settings: AzureOpenAISettings | None = None) -> None:
        self.settings = settings or AzureOpenAISettings.from_environment()
        self._client = AzureOpenAI(
            api_key=self.settings.api_key,
            azure_endpoint=self.settings.endpoint,
            api_version=self.settings.api_version,
        )

    def generate(self, user_input: str, *, system_prompt: str | None = None) -> str:
        """Generate one response using the configured Azure deployment."""

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_input})

        response = self._client.chat.completions.create(
            model=self.settings.deployment_name,
            messages=messages,
            temperature=self.settings.temperature,
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Azure OpenAI returned an empty response.")
        return content.strip()
