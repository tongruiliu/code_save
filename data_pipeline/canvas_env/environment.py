from __future__ import annotations

from typing import Any, Dict, Optional

from .blackboard import Blackboard
from .tool_registry import execute_tool, normalize_tool_call


class CanvasEnvironment:
    """Thin runtime wrapper over Blackboard + tool registry."""

    def __init__(self, initial_svg: Optional[str] = None):
        self.blackboard = Blackboard(initial_svg=initial_svg)

    def reset(self, initial_svg: Optional[str] = None) -> None:
        self.blackboard = Blackboard(initial_svg=initial_svg)

    def execute(self, tool_name: str, arguments: Dict[str, Any], render_path: Optional[str] = None) -> Dict[str, Any]:
        tool_result = execute_tool(self.blackboard, tool_name, arguments)
        render_result = None
        if render_path:
            render_result = self.blackboard.render_state(render_path)
        return {
            "tool_result": tool_result,
            "render_result": render_result,
        }

    def execute_tool_call(self, tool_call: Dict[str, Any], render_path: Optional[str] = None) -> Dict[str, Any]:
        tool_name, arguments = normalize_tool_call(tool_call)
        return self.execute(tool_name=tool_name, arguments=arguments, render_path=render_path)
