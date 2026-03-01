from __future__ import annotations

import base64
import json
import mimetypes
import os
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Dict, List, Optional


@dataclass
class TaskItem:
    task_id: int
    pid: str
    question: str
    instruction: str
    image_path: str
    answers: List[str]


def path_to_data_url(path_or_url: str) -> str:
    v = str(path_or_url or "").strip()
    if not v:
        return ""
    if v.startswith(("data:", "http://", "https://")):
        return v
    path = v[7:] if v.startswith("file://") else v
    if not os.path.exists(path):
        return ""
    mime, _ = mimetypes.guess_type(path)
    if not mime:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".svg":
            mime = "image/svg+xml"
        elif ext in {".jpg", ".jpeg"}:
            mime = "image/jpeg"
        elif ext == ".webp":
            mime = "image/webp"
        else:
            mime = "image/png"
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{encoded}"


def _coerce_items(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        if isinstance(payload.get("data"), list):
            return [x for x in payload["data"] if isinstance(x, dict)]
        kv = [(str(k), v) for k, v in payload.items() if isinstance(v, dict)]
        kv.sort(key=lambda x: (0, f"{int(x[0]):012d}") if x[0].isdigit() else (1, x[0]))
        return [v for _, v in kv]
    return []


def _choices_text(choices: Any) -> str:
    if not isinstance(choices, list) or not choices:
        return ""
    lines = []
    for i, c in enumerate(choices):
        label = chr(ord("A") + i) if i < 26 else f"C{i+1}"
        lines.append(f"{label}. {c}")
    return "Choices:\n" + "\n".join(lines)


def load_mathvista_tasks(
    json_path: str,
    data_root: Optional[str],
    start: int,
    end: int,
    skip_answer_type_list: bool = True,
) -> List[TaskItem]:
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    items = _coerce_items(payload)
    if start < 0:
        start = 0
    if end > len(items):
        end = len(items)
    if end <= start:
        return []

    root = data_root or os.path.dirname(json_path)
    selected = items[start:end]
    tasks: List[TaskItem] = []

    for i, item in enumerate(selected):
        q = str(item.get("question", "")).strip()
        if not q:
            continue
        answer_type = str(item.get("answer_type", "")).strip().lower()
        if skip_answer_type_list and answer_type == "list":
            continue

        image_rel = str(item.get("image", "")).strip()
        image_path = image_rel if os.path.isabs(image_rel) else os.path.join(root, image_rel)

        choices = _choices_text(item.get("choices"))
        instruction = q if not choices else f"{q}\n{choices}"

        ans = item.get("answer")
        answers: List[str] = []
        if isinstance(ans, list):
            answers.append(json.dumps(ans, ensure_ascii=False))
            answers.extend(str(x) for x in ans)
        elif ans is not None:
            s = str(ans).strip()
            if s:
                answers.append(s)

        pid = str(item.get("pid") or item.get("id") or f"sample_{start+i}")
        tasks.append(
            TaskItem(
                task_id=start + i,
                pid=pid,
                question=q,
                instruction=instruction,
                image_path=image_path,
                answers=answers,
            )
        )
    return tasks


def normalize_answer(s: str) -> str:
    return "".join(str(s or "").strip().lower().split())


def parse_numeric(s: str) -> Optional[float]:
    t = normalize_answer(s)
    if not t:
        return None
    try:
        if "/" in t and t.count("/") == 1:
            return float(Fraction(t))
        return float(t)
    except Exception:
        return None


def equivalent_answer(model_answer: str, gold_answer: str) -> bool:
    m = normalize_answer(model_answer)
    g = normalize_answer(gold_answer)
    if not m or not g:
        return False
    if m == g:
        return True
    mn = parse_numeric(m)
    gn = parse_numeric(g)
    if mn is not None and gn is not None:
        return abs(mn - gn) <= 1e-9
    return False
