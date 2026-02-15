from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable


class CanvasTool(ABC):
    """Tool interface for Canvas environment operations."""

    name: str = ""
    description: str = ""
    required_fields: Iterable[str] = ()

    @classmethod
    @abstractmethod
    def parameter_schema(cls) -> Dict[str, Any]:
        """Return OpenAI function-calling parameter schema."""

    @classmethod
    def to_schema(cls) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": cls.name,
                "description": cls.description,
                "parameters": cls.parameter_schema(),
            },
        }

    @classmethod
    def normalize_arguments(cls, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if arguments is None:
            return {}
        if not isinstance(arguments, dict):
            raise TypeError(f"{cls.name} arguments must be an object")
        return dict(arguments)

    @classmethod
    def validate_arguments(cls, arguments: Dict[str, Any]) -> None:
        for key in cls.required_fields:
            if key not in arguments:
                raise ValueError(f"{cls.name} missing required argument: {key}")

    @classmethod
    def execute(cls, blackboard: Any, arguments: Dict[str, Any]) -> Dict[str, Any]:
        normalized = cls.normalize_arguments(arguments)
        cls.validate_arguments(normalized)
        blackboard.update_state(action=cls.name, attrs=normalized)
        return normalized
