from __future__ import annotations

from typing import Any, Dict, List, Tuple, Type

from .tools import (
    CanvasTool,
    ClearTool,
    InsertElementTool,
    ModifyElementTool,
    RemoveElementTool,
    ReplaceElementTool,
)

TOOL_CLASSES: Tuple[Type[CanvasTool], ...] = (
    InsertElementTool,
    ModifyElementTool,
    RemoveElementTool,
    ReplaceElementTool,
    ClearTool,
)

TOOL_NAME_TO_CLASS: Dict[str, Type[CanvasTool]] = {tool.name: tool for tool in TOOL_CLASSES}
SUPPORTED_TOOLS: List[str] = list(TOOL_NAME_TO_CLASS.keys())


def get_tool_schemas() -> List[Dict[str, Any]]:
    return [tool.to_schema() for tool in TOOL_CLASSES]


def get_tool_class(tool_name: str) -> Type[CanvasTool]:
    if tool_name not in TOOL_NAME_TO_CLASS:
        raise KeyError(f"Unknown tool: {tool_name}")
    return TOOL_NAME_TO_CLASS[tool_name]


def normalize_tool_call(tool_call: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    if not isinstance(tool_call, dict):
        raise TypeError("tool_call must be an object")
    if "name" not in tool_call:
        raise ValueError("tool_call missing 'name'")
    name = tool_call["name"]
    arguments = tool_call.get("arguments", {})
    if not isinstance(name, str) or not name:
        raise ValueError("tool_call.name must be non-empty string")
    if not isinstance(arguments, dict):
        raise ValueError("tool_call.arguments must be an object")
    return name, arguments


def execute_tool(blackboard: Any, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    tool_cls = get_tool_class(tool_name)
    normalized_args = tool_cls.execute(blackboard=blackboard, arguments=arguments)
    return {
        "tool": tool_name,
        "arguments": normalized_args,
    }
