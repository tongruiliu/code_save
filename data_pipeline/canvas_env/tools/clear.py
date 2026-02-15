from __future__ import annotations

from typing import Any, Dict

from .base import CanvasTool


class ClearTool(CanvasTool):
    name = "clear"
    description = "Clear all elements. Use only for full reset."
    required_fields = ()

    @classmethod
    def parameter_schema(cls) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
        }
