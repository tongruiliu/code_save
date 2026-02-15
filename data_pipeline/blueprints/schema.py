from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List

from ..canvas_env.tool_registry import SUPPORTED_TOOLS


@dataclass
class ValidationResult:
    valid: bool
    errors: List[str]


BASELINE_SAFE_IDS = {"root", "main_svg"}
BASE_IMAGE_ID = "base_image"
DISALLOWED_ATTR_NAMES = {"text", "direction"}
PLACEHOLDER_VALUE_PATTERNS = [
    r"\bpath data\b",
    r"\baligned with\b",
    r"\bmatches\b",
    r"\bmatch\b",
    r"\bangular position\b",
    r"\bcontact point\b",
    r"\bplaceholder\b",
    r"\betc\b",
]


def extract_json_block(text: str) -> str:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return text
    return m.group(0)


def _extract_ids_from_init_svg(init_svg: str) -> List[str]:
    if not isinstance(init_svg, str):
        return []
    return re.findall(r"id\s*=\s*['\"]([^'\"]+)['\"]", init_svg)


def _has_svg_id(init_svg: str, element_id: str) -> bool:
    pattern = re.compile(rf"id\s*=\s*['\"]{re.escape(element_id)}['\"]", flags=re.IGNORECASE)
    return pattern.search(init_svg or "") is not None


def _looks_placeholder_value(value: str) -> bool:
    v = (value or "").strip().lower()
    if not v:
        return True
    for pat in PLACEHOLDER_VALUE_PATTERNS:
        if re.search(pat, v):
            return True
    return False


def validate_blueprint(bp: Dict[str, Any]) -> ValidationResult:
    errors: List[str] = []

    def _need_str(key: str) -> None:
        if not isinstance(bp.get(key), str) or not bp[key].strip():
            errors.append(f"{key} must be non-empty string")

    for field in ("task_id", "question", "image_path", "init_svg"):
        _need_str(field)

    reveal_plan = bp.get("user_reveal_plan")
    if not isinstance(reveal_plan, list) or not (2 <= len(reveal_plan) <= 6):
        errors.append("user_reveal_plan must be list with length in [2, 6]")
    else:
        if not all(isinstance(x, str) and x.strip() for x in reveal_plan):
            errors.append("user_reveal_plan must contain non-empty strings")

    constraints = bp.get("constraints")
    allowed_tools: List[str] = []
    if not isinstance(constraints, dict):
        errors.append("constraints must be object")
    else:
        max_turns = constraints.get("max_turns")
        if not isinstance(max_turns, int) or not (4 <= max_turns <= 20):
            errors.append("constraints.max_turns must be int in [4, 20]")
        tools = constraints.get("allowed_tools")
        if not isinstance(tools, list) or not tools:
            errors.append("constraints.allowed_tools must be non-empty list")
        else:
            allowed_tools = [t for t in tools if isinstance(t, str)]
            invalid_tools = [t for t in allowed_tools if t not in SUPPORTED_TOOLS]
            if invalid_tools:
                errors.append(f"constraints.allowed_tools has unsupported tools: {invalid_tools}")

    raw_plan = bp.get("optional_gt_action_plan", None)
    if raw_plan is not None:
        if not isinstance(raw_plan, list) or not raw_plan:
            errors.append("optional_gt_action_plan must be non-empty list when provided")
        else:
            for i, item in enumerate(raw_plan):
                if not isinstance(item, dict):
                    errors.append(f"optional_gt_action_plan[{i}] must be object")
                    continue
                name = item.get("name", item.get("tool", item.get("action", "")))
                if not isinstance(name, str) or not name.strip():
                    errors.append(f"optional_gt_action_plan[{i}].name must be non-empty string")
                    continue
                if name not in SUPPORTED_TOOLS:
                    errors.append(f"optional_gt_action_plan[{i}] has unsupported tool: {name}")
                if allowed_tools and name not in allowed_tools:
                    errors.append(
                        f"optional_gt_action_plan[{i}] tool '{name}' not in constraints.allowed_tools"
                    )
                args = item.get("arguments", item.get("args", item.get("kwargs", {})))
                if not isinstance(args, dict):
                    errors.append(f"optional_gt_action_plan[{i}].arguments must be object")

    init_svg = str(bp.get("init_svg", ""))
    if not _has_svg_id(init_svg, "main_svg"):
        errors.append("init_svg must contain editable svg id `main_svg`")
    if not _has_svg_id(init_svg, BASE_IMAGE_ID):
        errors.append("init_svg must include input image background id `base_image`")
    baseline_ids = set(_extract_ids_from_init_svg(init_svg)) | BASELINE_SAFE_IDS

    success = bp.get("success_check")
    if not isinstance(success, dict):
        errors.append("success_check must be object")
    else:
        ids = success.get("must_have_ids")
        if not isinstance(ids, list) or not all(isinstance(x, str) for x in ids):
            errors.append("success_check.must_have_ids must be list[str]")
            ids = []
        else:
            ids = [x.strip() for x in ids if isinstance(x, str) and x.strip()]
            if len(ids) != len(set(ids)):
                errors.append("success_check.must_have_ids must not contain duplicates")
            if any(x == BASE_IMAGE_ID for x in ids):
                errors.append("success_check.must_have_ids must not target base image id `base_image`")
        attrs = success.get("must_have_attrs")
        if not isinstance(attrs, list):
            errors.append("success_check.must_have_attrs must be list")
            attrs = []
        else:
            for i, item in enumerate(attrs):
                if not isinstance(item, dict):
                    errors.append(f"success_check.must_have_attrs[{i}] must be object")
                    continue
                for key in ("id", "attr", "value"):
                    if not isinstance(item.get(key), str):
                        errors.append(f"success_check.must_have_attrs[{i}].{key} must be string")
                item_id = str(item.get("id", "")).strip()
                item_attr = str(item.get("attr", "")).strip()
                item_value = str(item.get("value", "")).strip()
                if not item_id:
                    errors.append(f"success_check.must_have_attrs[{i}].id must be non-empty")
                if not item_attr:
                    errors.append(f"success_check.must_have_attrs[{i}].attr must be non-empty")
                if item_id == BASE_IMAGE_ID:
                    errors.append(
                        f"success_check.must_have_attrs[{i}] must not target base image id `base_image`"
                    )
                if item_attr.lower() in DISALLOWED_ATTR_NAMES:
                    errors.append(
                        f"success_check.must_have_attrs[{i}].attr uses unsupported pseudo-attribute: {item_attr}"
                    )
                if _looks_placeholder_value(item_value):
                    errors.append(
                        f"success_check.must_have_attrs[{i}].value looks non-deterministic placeholder: {item_value}"
                    )
        if not isinstance(success.get("need_boxed_answer"), bool):
            errors.append("success_check.need_boxed_answer must be boolean")

        # Semantic checkability: require at least one post-edit target beyond init canvas baseline.
        attr_ids = []
        if isinstance(attrs, list):
            for item in attrs:
                if isinstance(item, dict) and isinstance(item.get("id"), str):
                    attr_ids.append(item.get("id", "").strip())
        target_ids = set([x for x in ids if x] + [x for x in attr_ids if x])
        non_baseline_targets = [x for x in target_ids if x not in baseline_ids]
        if not non_baseline_targets:
            errors.append(
                "success_check must include at least one target id beyond init_svg baseline "
                "(otherwise stage-2 can pass without meaningful canvas edits)"
            )

        if non_baseline_targets and isinstance(allowed_tools, list):
            if not any(t in {"insert_element", "replace_element"} for t in allowed_tools):
                errors.append(
                    "constraints.allowed_tools must include insert_element or replace_element "
                    "when success_check requires new canvas targets"
                )

    return ValidationResult(valid=len(errors) == 0, errors=errors)
