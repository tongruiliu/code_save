from __future__ import annotations

from typing import Any, Dict

from .base import CanvasTool


class RemoveElementTool(CanvasTool):
    name = "remove_element"
    description = "Remove an element from the blackboard."
    required_fields = ("targetId",)

    @classmethod
    def parameter_schema(cls) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "targetId": {
                    "type": "string",
                    "description": "The id of the element to remove.",
                }
            },
            "required": ["targetId"],
        }
