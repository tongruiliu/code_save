from __future__ import annotations

import json
import re
from typing import Any, Dict, List


THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
TOOL_PATTERN = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
BOXED_PATTERN = re.compile(r"\\boxed\{.*?\}", re.DOTALL)
ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)


def parse_model_response(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    if raw.count("<think>") == 1 and raw.count("</think>") == 0:
        idx = raw.find("<tool_call>")
        if idx != -1:
            raw = raw[:idx] + "</think>" + raw[idx:]

    think_match = THINK_PATTERN.search(raw)
    thought = think_match.group(1).strip() if think_match else ""

    tool_calls: List[Dict[str, Any]] = []
    for m in TOOL_PATTERN.finditer(raw):
        chunk = m.group(1).strip().replace("\\n", "\\\\n")
        try:
            payload = json.loads(chunk)
            if isinstance(payload, dict) and "name" in payload and "arguments" in payload:
                tool_calls.append(payload)
        except Exception:
            continue

    clean_text = THINK_PATTERN.sub("", raw)
    clean_text = TOOL_PATTERN.sub("", clean_text).strip()
    has_boxed = BOXED_PATTERN.search(raw) is not None
    answer_blocks = [m.group(1).strip() for m in ANSWER_PATTERN.finditer(raw)]
    has_answer_tag = bool(answer_blocks)
    has_boxed_in_answer = any(BOXED_PATTERN.search(block) is not None for block in answer_blocks)
    has_final_answer = has_answer_tag and has_boxed_in_answer

    return {
        "raw": raw,
        "thought": thought,
        "tool_calls": tool_calls,
        "text": clean_text,
        "has_boxed": has_boxed,
        "has_answer_tag": has_answer_tag,
        "has_boxed_in_answer": has_boxed_in_answer,
        "has_final_answer": has_final_answer,
    }
