from .tool_registry import SUPPORTED_TOOLS, execute_tool, get_tool_schemas

try:
    from .blackboard import Blackboard
    from .environment import CanvasEnvironment
except Exception:  # Optional runtime deps (bs4/playwright) may be missing.
    Blackboard = None  # type: ignore
    CanvasEnvironment = None  # type: ignore

TOOL_SCHEMAS = get_tool_schemas()

__all__ = ["Blackboard", "CanvasEnvironment", "SUPPORTED_TOOLS", "TOOL_SCHEMAS", "get_tool_schemas", "execute_tool"]
