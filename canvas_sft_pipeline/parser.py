from __future__ import annotations

import json
import re
from typing import Any, Dict, List


THINK_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.IGNORECASE | re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
BOXED_RE = re.compile(r"(?:\\boxed|/boxed)\{(.*)\}", re.DOTALL)


def _close_unpaired_think(raw: str) -> str:
    text = str(raw or "")
    if text.count("<think>") == 1 and text.count("</think>") == 0:
        idx = text.find("<tool_call>")
        if idx != -1:
            text = text[:idx] + "</think>" + text[idx:]
    return text


def _try_json_tool(raw: str) -> Dict[str, Any] | None:
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        # Keep compatibility with Canvas parsing for escaped newlines in tool args.
        try:
            obj = json.loads(raw.replace("\\n", "\\\\n"))
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
    raw = _close_unpaired_think(text)

    think_match = THINK_RE.search(raw)
    reasoning_content = think_match.group(1).strip() if think_match else ""

    tool_calls: List[Dict[str, Any]] = []
    malformed_tool_calls = 0
    for m in TOOL_CALL_RE.finditer(raw):
        block = m.group(1).strip()
        obj = _try_json_tool(block)
        if obj is None:
            malformed_tool_calls += 1
        else:
            tool_calls.append(obj)

    answer = ""
    ans_match = ANSWER_RE.search(raw)
    if ans_match:
        answer = ans_match.group(1).strip()

    content = THINK_RE.sub("", raw)
    content = TOOL_CALL_RE.sub("", content).strip()

    seed_content = raw.split("</think>")[-1].strip() if "</think>" in raw else content
    if not seed_content:
        seed_content = reasoning_content

    boxed = ""
    boxed_match = BOXED_RE.search(answer or content or raw)
    if boxed_match:
        boxed = boxed_match.group(1).strip()

    return {
        "raw": raw,
        "reasoning_content": reasoning_content,
        "think": reasoning_content,
        "tool_calls": tool_calls,
        "malformed_tool_calls": malformed_tool_calls,
        "answer": answer,
        "boxed": boxed,
        "content": content,
        "seed_content": seed_content,
        "has_answer_tag": "<answer>" in raw.lower(),
    }

