from __future__ import annotations

from typing import Any, Dict

from .base import CanvasTool


class InsertElementTool(CanvasTool):
    name = "insert_element"
    description = "Insert a new SVG or HTML element into the blackboard."
    required_fields = ("fragment", "rootId")

    @classmethod
    def parameter_schema(cls) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "fragment": {
                    "type": "string",
                    "description": "The HTML/SVG string to insert.",
                },
                "rootId": {
                    "type": "string",
                    "description": "ID of parent container to append to.",
                },
                "beforeId": {
                    "type": "string",
                    "description": "Optional sibling id to insert before.",
                },
            },
            "required": ["fragment", "rootId"],
        }
