from __future__ import annotations

from typing import Any

from anthropic import Anthropic

from providers.base import LLMProvider


class ClaudeProvider(LLMProvider):
    provider_name = "anthropic"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-haiku-4-5",
        max_tokens: int = 16384,
    ):
        # Anthropic SDK can also read ANTHROPIC_API_KEY from the environment
        self.client = Anthropic(api_key=api_key) if api_key else Anthropic()
        self.model = model
        self.max_tokens = max_tokens

    def _call_model(
        self, system_prompt: str, user_prompt: str, temperature: float
    ) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self._clamp_temperature(temperature),
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ],
        )
        return self._extract_response_text(response)

    def _call_repair_model(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
    ) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=0.0,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ],
        )
        return self._extract_response_text(response)

    @staticmethod
    def _clamp_temperature(value: float) -> float:
        try:
            value = float(value)
        except Exception:
            value = 0.2
        return max(0.0, min(1.0, value))

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        parts: list[str] = []

        for block in getattr(response, "content", []) or []:
            block_type = getattr(block, "type", None)

            if block_type == "text":
                text = getattr(block, "text", None)
                if isinstance(text, str) and text.strip():
                    parts.append(text)

            elif isinstance(block, dict):
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    text = block["text"].strip()
                    if text:
                        parts.append(text)

        merged = "\n".join(parts).strip()
        if merged:
            return merged

        raise ValueError("Claude returned an empty response; expected text/JSON.")
