from __future__ import annotations

from typing import Any, Dict, List


blackboard_tools: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "insert_element",
            "description": "Insert a new SVG or HTML element into the blackboard.",
            "parameters": {
                "type": "object",
                "properties": {
                    "fragment": {"type": "string"},
                    "rootId": {"type": "string"},
                    "beforeId": {"type": ["string", "null"]},
                },
                "required": ["fragment", "rootId"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "modify_element",
            "description": "Update attributes of an existing element.",
            "parameters": {
                "type": "object",
                "properties": {
                    "targetId": {"type": "string"},
                    "attrs": {"type": "object", "additionalProperties": True},
                },
                "required": ["targetId", "attrs"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remove_element",
            "description": "Remove an element from the blackboard.",
            "parameters": {
                "type": "object",
                "properties": {"targetId": {"type": "string"}},
                "required": ["targetId"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_element",
            "description": "Replace an existing element with a new fragment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "targetId": {"type": "string"},
                    "fragment": {"type": "string"},
                },
                "required": ["targetId", "fragment"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "clear",
            "description": "Clear all elements.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]
