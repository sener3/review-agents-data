from __future__ import annotations

from openai import OpenAI

from providers.base import LLMProvider


class OpenAIProvider(LLMProvider):
    provider_name = "openai"

    def __init__(self, api_key: str, model: str = "gpt-5-mini"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def _call_model(
        self, system_prompt: str, user_prompt: str, temperature: float
    ) -> str:
        kwargs = {
            "model": self.model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        if self.model.startswith("gpt-5"):
            kwargs["reasoning"] = {
                "effort": self._temperature_to_reasoning_effort(temperature)
            }
        else:
            kwargs["temperature"] = temperature

        response = self.client.responses.create(**kwargs)
        return response.output_text or ""

    def _call_repair_model(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
    ) -> str:
        kwargs = {
            "model": self.model,
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        if self.model.startswith("gpt-5"):
            kwargs["reasoning"] = {"effort": "minimal"}
        else:
            kwargs["temperature"] = 0.0

        response = self.client.responses.create(**kwargs)
        return response.output_text or ""

    @staticmethod
    def _temperature_to_reasoning_effort(temperature: float) -> str:
        try:
            t = float(temperature)
        except Exception:
            t = 0.2

        if t <= 0.10:
            return "minimal"
        if t <= 0.30:
            return "low"
        if t <= 0.60:
            return "medium"
        return "high"
