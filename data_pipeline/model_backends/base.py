from __future__ import annotations


class ModelBackend:
    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.0) -> str:
        raise NotImplementedError
