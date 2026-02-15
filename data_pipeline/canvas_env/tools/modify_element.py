from __future__ import annotations

from typing import Any, Dict

from .base import CanvasTool


class ModifyElementTool(CanvasTool):
    name = "modify_element"
    description = "Update attributes of an existing element."
    required_fields = ("targetId", "attrs")

    @classmethod
    def parameter_schema(cls) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "targetId": {
                    "type": "string",
                    "description": "The id of the element to modify.",
                },
                "attrs": {
                    "type": "object",
                    "description": "Key-value attributes to update.",
                    "additionalProperties": True,
                },
            },
            "required": ["targetId", "attrs"],
        }
