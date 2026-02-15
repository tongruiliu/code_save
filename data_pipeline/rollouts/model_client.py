from __future__ import annotations

import base64
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..model_backends import ApiBackend, LocalBackend


def image_path_to_data_url(path: str) -> str:
    p = Path(path)
    data = p.read_bytes()
    suffix = p.suffix.lower().lstrip(".") or "png"
    return f"data:image/{suffix};base64,{base64.b64encode(data).decode('utf-8')}"


class ChatClient:
    supports_multimodal: bool = False

    def generate(self, messages: List[Dict[str, Any]], temperature: float, max_tokens: int) -> str:
        raise NotImplementedError


class ApiChatClient(ChatClient):
    supports_multimodal = True

    def __init__(self, model: str, api_key: str, base_url: Optional[str] = None, timeout: int = 120):
        self.backend = ApiBackend(model=model, api_key=api_key, base_url=base_url, timeout=timeout)

    def generate(self, messages: List[Dict[str, Any]], temperature: float, max_tokens: int) -> str:
        resp = self.backend.client.chat.completions.create(
            model=self.backend.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = resp.choices[0].message.content
        return content if content is not None else ""


class LocalChatClient(ChatClient):
    supports_multimodal = False

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        max_new_tokens: int = 1024,
        trust_remote_code: bool = False,
    ):
        self.backend = LocalBackend(
            model_path=model_path,
            device=device,
            max_new_tokens=max_new_tokens,
            trust_remote_code=trust_remote_code,
        )

    @staticmethod
    def _flatten_messages(messages: List[Dict[str, Any]]) -> str:
        lines: List[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                parts: List[str] = []
                for item in content:
                    if item.get("type") == "text":
                        parts.append(item.get("text", ""))
                    elif item.get("type") == "image_url":
                        parts.append("<image>")
                text = "\n".join(parts)
            else:
                text = str(content)
            lines.append(f"[{role}]\n{text}")
        return "\n\n".join(lines)

    def generate(self, messages: List[Dict[str, Any]], temperature: float, max_tokens: int) -> str:
        flat = self._flatten_messages(messages)
        return self.backend.generate(
            system_prompt="You are a helpful assistant.",
            user_prompt=flat,
            temperature=temperature,
        )
