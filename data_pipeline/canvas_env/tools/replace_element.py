from __future__ import annotations

from typing import Any, Dict

from .base import CanvasTool


class ReplaceElementTool(CanvasTool):
    name = "replace_element"
    description = "Completely replace an existing element with a new code fragment."
    required_fields = ("targetId", "fragment")

    @classmethod
    def parameter_schema(cls) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "targetId": {
                    "type": "string",
                    "description": "The id of element to replace.",
                },
                "fragment": {
                    "type": "string",
                    "description": "New HTML/SVG fragment.",
                },
            },
            "required": ["targetId", "fragment"],
        }
