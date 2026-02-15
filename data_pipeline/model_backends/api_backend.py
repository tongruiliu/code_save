from __future__ import annotations

from typing import Any, Dict, Optional

from .base import ModelBackend


class ApiBackend(ModelBackend):
    def __init__(self, model: str, api_key: str, base_url: Optional[str], timeout: int = 120):
        try:
            from openai import OpenAI
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("openai package is required for API backend") from exc

        self.model = model
        kwargs: Dict[str, Any] = {"api_key": api_key, "timeout": timeout}
        if base_url:
            kwargs["base_url"] = base_url
        self.client = OpenAI(**kwargs)

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
        content = resp.choices[0].message.content
        return content if content is not None else ""
