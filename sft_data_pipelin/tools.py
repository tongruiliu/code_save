from __future__ import annotations

import copy
import json
import re
from typing import Any, Dict, List


def init_canvas_data() -> Dict[str, Any]:
    return {
        "canvas": {
            "nodes": {
                "root": {
                    "id": "root",
                    "tag": "div",
                    "fragment": '<div id="root"></div>',
                    "parent": None,
                    "children": [],
                    "attrs": {},
                    "text": "",
                }
            }
        }
    }


def _extract_tag(fragment: str) -> str:
    m = re.search(r"<\s*([a-zA-Z0-9_:-]+)", fragment)
    return m.group(1) if m else "unknown"


def _extract_id(fragment: str) -> str | None:
    m = re.search(r"\bid\s*=\s*[\"']([^\"']+)[\"']", fragment)
    return m.group(1) if m else None


def _get_nodes(data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    if "canvas" not in data:
        data.update(init_canvas_data())
    if "nodes" not in data["canvas"]:
        data["canvas"]["nodes"] = init_canvas_data()["canvas"]["nodes"]
    return data["canvas"]["nodes"]


def _remove_subtree(nodes: Dict[str, Dict[str, Any]], node_id: str) -> None:
    for child_id in list(nodes[node_id]["children"]):
        _remove_subtree(nodes, child_id)
    parent_id = nodes[node_id]["parent"]
    if parent_id and node_id in nodes[parent_id]["children"]:
        nodes[parent_id]["children"].remove(node_id)
    del nodes[node_id]


def canvas_snapshot(data: Dict[str, Any]) -> Dict[str, Any]:
    nodes = _get_nodes(data)

    def build(node_id: str) -> Dict[str, Any]:
        node = nodes[node_id]
        return {
            "id": node["id"],
            "tag": node["tag"],
            "attrs": copy.deepcopy(node["attrs"]),
            "text": node.get("text", ""),
            "children": [build(cid) for cid in node["children"]],
        }

    return build("root")


class Tool:
    @staticmethod
    def invoke(*args: Any, **kwargs: Any) -> str:
        raise NotImplementedError

    @staticmethod
    def get_info() -> Dict[str, Any]:
        raise NotImplementedError


class InsertElement(Tool):
    @staticmethod
    def invoke(
        data: Dict[str, Any],
        fragment: str,
        rootId: str = "root",
        beforeId: str | None = None,
    ) -> str:
        nodes = _get_nodes(data)
        if rootId not in nodes:
            return f"Error: rootId '{rootId}' not found"

        node_id = _extract_id(fragment)
        if node_id is None:
            return "Error: fragment must include an id attribute"
        if node_id in nodes:
            return f"Error: id '{node_id}' already exists"

        new_node = {
            "id": node_id,
            "tag": _extract_tag(fragment),
            "fragment": fragment,
            "parent": rootId,
            "children": [],
            "attrs": {},
            "text": "",
        }
        nodes[node_id] = new_node

        children = nodes[rootId]["children"]
        if beforeId is not None:
            if beforeId not in children:
                del nodes[node_id]
                return f"Error: beforeId '{beforeId}' is not a child of '{rootId}'"
            idx = children.index(beforeId)
            children.insert(idx, node_id)
        else:
            children.append(node_id)

        return json.dumps({
            "status": "ok",
            "action": "insert_element",
            "inserted_id": node_id,
            "canvas": canvas_snapshot(data),
        }, ensure_ascii=False)

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "insert_element",
                "description": "Insert a new HTML/SVG element into the canvas tree.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "fragment": {
                            "type": "string",
                            "description": "HTML/SVG fragment to insert. Must include an id attribute.",
                        },
                        "rootId": {
                            "type": "string",
                            "description": "Parent node id. Use 'root' for top-level insert.",
                            "default": "root",
                        },
                        "beforeId": {
                            "type": ["string", "null"],
                            "description": "Insert before this sibling id (must be child of rootId).",
                            "default": None,
                        },
                    },
                    "required": ["fragment", "rootId"],
                },
            },
        }


class ModifyElement(Tool):
    @staticmethod
    def invoke(data: Dict[str, Any], targetId: str, attrs: Dict[str, Any]) -> str:
        nodes = _get_nodes(data)
        if targetId not in nodes:
            return f"Error: targetId '{targetId}' not found"
        node = nodes[targetId]
        for k, v in attrs.items():
            if k == "text":
                node["text"] = str(v)
            else:
                node["attrs"][str(k)] = v

        return json.dumps({
            "status": "ok",
            "action": "modify_element",
            "target_id": targetId,
            "canvas": canvas_snapshot(data),
        }, ensure_ascii=False)

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "modify_element",
                "description": "Update attributes of an existing element.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "targetId": {"type": "string"},
                        "attrs": {
                            "type": "object",
                            "additionalProperties": True,
                            "description": "Attribute map, e.g. {'fill':'#1377EB','text':'new value'}",
                        },
                    },
                    "required": ["targetId", "attrs"],
                },
            },
        }


class ReplaceElement(Tool):
    @staticmethod
    def invoke(data: Dict[str, Any], targetId: str, fragment: str) -> str:
        nodes = _get_nodes(data)
        if targetId not in nodes:
            return f"Error: targetId '{targetId}' not found"

        old_node = nodes[targetId]
        parent_id = old_node["parent"]
        if parent_id is None:
            return "Error: root cannot be replaced"

        new_id = _extract_id(fragment) or targetId
        if new_id != targetId and new_id in nodes:
            return f"Error: id '{new_id}' already exists"

        siblings = nodes[parent_id]["children"]
        idx = siblings.index(targetId)

        replacement = {
            "id": new_id,
            "tag": _extract_tag(fragment),
            "fragment": fragment,
            "parent": parent_id,
            "children": old_node["children"],
            "attrs": {},
            "text": "",
        }

        del nodes[targetId]
        nodes[new_id] = replacement
        siblings[idx] = new_id

        for child_id in replacement["children"]:
            nodes[child_id]["parent"] = new_id

        return json.dumps({
            "status": "ok",
            "action": "replace_element",
            "target_id": targetId,
            "new_id": new_id,
            "canvas": canvas_snapshot(data),
        }, ensure_ascii=False)

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {
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
        }


class RemoveElement(Tool):
    @staticmethod
    def invoke(data: Dict[str, Any], targetId: str) -> str:
        nodes = _get_nodes(data)
        if targetId == "root":
            return "Error: root cannot be removed"
        if targetId not in nodes:
            return f"Error: targetId '{targetId}' not found"

        _remove_subtree(nodes, targetId)
        return json.dumps({
            "status": "ok",
            "action": "remove_element",
            "removed_id": targetId,
            "canvas": canvas_snapshot(data),
        }, ensure_ascii=False)

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "remove_element",
                "description": "Remove an element (and its subtree) from the canvas.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "targetId": {"type": "string"},
                    },
                    "required": ["targetId"],
                },
            },
        }


class ClearCanvas(Tool):
    @staticmethod
    def invoke(data: Dict[str, Any]) -> str:
        data.clear()
        data.update(init_canvas_data())
        return json.dumps({
            "status": "ok",
            "action": "clear",
            "canvas": canvas_snapshot(data),
        }, ensure_ascii=False)

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "clear",
                "description": "Clear all elements under root.",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            },
        }


class FinishCanvas(Tool):
    @staticmethod
    def invoke(data: Dict[str, Any], summary: str = "") -> str:
        return json.dumps({"status": "ok", "action": "finish_canvas", "summary": summary}, ensure_ascii=False)

    @staticmethod
    def get_info() -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": "finish_canvas",
                "description": "Call this when all requested canvas operations are complete.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                    },
                    "required": [],
                },
            },
        }


ALL_TOOLS = [
    InsertElement,
    ModifyElement,
    ReplaceElement,
    RemoveElement,
    ClearCanvas,
    FinishCanvas,
]
