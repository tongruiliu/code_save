from __future__ import annotations

import json
import re
from typing import Any, Dict, List


THINK_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.IGNORECASE | re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
BOXED_RE = re.compile(r"(?:\\boxed|/boxed)\{(.*)\}", re.DOTALL)


def _try_json(raw: str) -> Dict[str, Any] | None:
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    name = obj.get("name") or obj.get("tool")
    args = obj.get("arguments", obj.get("args", obj.get("kwargs", {})))
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return None
    if not isinstance(name, str) or not isinstance(args, dict):
        return None
    return {"name": name, "arguments": args}


def parse_response(text: str) -> Dict[str, Any]:
    raw = str(text or "")
    think = ""
    m = THINK_RE.search(raw)
    if m:
        think = m.group(1).strip()

    tool_calls: List[Dict[str, Any]] = []
    malformed_tool_calls = 0
    for m in TOOL_CALL_RE.finditer(raw):
        block = m.group(1).strip()
        obj = _try_json(block)
        if obj is None:
            malformed_tool_calls += 1
        else:
            tool_calls.append(obj)

    answer = ""
    m = ANSWER_RE.search(raw)
    if m:
        answer = m.group(1).strip()

    boxed = ""
    boxed_match = BOXED_RE.search(answer or raw)
    if boxed_match:
        boxed = boxed_match.group(1).strip()

    return {
        "raw": raw,
        "think": think,
        "tool_calls": tool_calls,
        "malformed_tool_calls": malformed_tool_calls,
        "answer": answer,
        "boxed": boxed,
        "has_answer_tag": "<answer>" in raw.lower(),
    }
