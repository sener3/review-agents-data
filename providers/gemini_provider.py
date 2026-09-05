from __future__ import annotations

from typing import Any

from google import genai
from google.genai import types

from providers.base import LLMProvider


class GeminiProvider(LLMProvider):
    provider_name = "google"

    def __init__(self, api_key: str | None = None, model: str = "gemini-2.5-flash"):
        self.client = genai.Client(api_key=api_key) if api_key else genai.Client()
        self.model = model

    def _call_model(
        self, system_prompt: str, user_prompt: str, temperature: float
    ) -> str:
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=self._clamp_temperature(temperature),
            response_mime_type="application/json",
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=config,
        )
        return self._extract_response_text(response)

    def _call_repair_model(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
    ) -> str:
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            temperature=0.0,
            response_mime_type="application/json",
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=config,
        )
        return self._extract_response_text(response)

    @staticmethod
    def _clamp_temperature(value: float) -> float:
        try:
            value = float(value)
        except Exception:
            value = 0.2
        return max(0.0, min(2.0, value))

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text

        try:
            parts: list[str] = []
            for candidate in getattr(response, "candidates", []) or []:
                content = getattr(candidate, "content", None)
                for part in getattr(content, "parts", []) or []:
                    part_text = getattr(part, "text", None)
                    if isinstance(part_text, str) and part_text.strip():
                        parts.append(part_text)

            merged = "\n".join(parts).strip()
            if merged:
                return merged
        except Exception:
            pass

        raise ValueError("Gemini returned an empty response; expected JSON.")
