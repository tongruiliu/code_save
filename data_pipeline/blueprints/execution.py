from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..canvas_env import CanvasEnvironment, SUPPORTED_TOOLS


@dataclass
class ExecutionValidationResult:
    valid: bool
    errors: List[str]
    summary: Dict[str, Any]


BASELINE_SAFE_IDS = {"root", "main_svg"}


def _id_exists(html: str, element_id: str) -> bool:
    pattern = re.compile(rf"id\s*=\s*['\"]{re.escape(element_id)}['\"]")
    return pattern.search(html) is not None


def _attr_equals(html: str, element_id: str, attr: str, value: str) -> bool:
    tag_pattern = re.compile(rf"<[^>]*id\s*=\s*['\"]{re.escape(element_id)}['\"][^>]*>")
    m = tag_pattern.search(html)
    if not m:
        return False
    tag = m.group(0)
    attr_pattern = re.compile(rf"{re.escape(attr)}\s*=\s*['\"]{re.escape(value)}['\"]")
    return attr_pattern.search(tag) is not None


def _extract_ids_from_svg(html_or_svg: str) -> List[str]:
    return re.findall(r"id\s*=\s*['\"]([^'\"]+)['\"]", html_or_svg or "")


def _extract_ids_from_fragment(fragment: str) -> List[str]:
    return re.findall(r"id\s*=\s*['\"]([^'\"]+)['\"]", fragment or "")


def _normalize_action_item(item: Dict[str, Any], idx: int) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not isinstance(item, dict):
        return None, f"optional_gt_action_plan[{idx}] must be object"

    name = item.get("name")
    if not isinstance(name, str) or not name.strip():
        name = item.get("tool")
    if not isinstance(name, str) or not name.strip():
        name = item.get("action")
    if not isinstance(name, str) or not name.strip():
        return None, f"optional_gt_action_plan[{idx}].name must be non-empty string"
    name = name.strip()

    arguments = item.get("arguments")
    if arguments is None:
        arguments = item.get("args")
    if arguments is None:
        arguments = item.get("kwargs")
    if arguments is None:
        arguments = {}

    if not isinstance(arguments, dict):
        return None, f"optional_gt_action_plan[{idx}].arguments must be object"

    return {"name": name, "arguments": arguments}, None


def _normalize_action_plan(raw_plan: Any) -> Tuple[List[Dict[str, Any]], List[str]]:
    if raw_plan is None:
        return [], []
    if not isinstance(raw_plan, list):
        return [], ["optional_gt_action_plan must be a list when provided"]

    normalized: List[Dict[str, Any]] = []
    errors: List[str] = []
    for i, item in enumerate(raw_plan):
        parsed, err = _normalize_action_item(item, i)
        if err:
            errors.append(err)
            continue
        if parsed is not None:
            normalized.append(parsed)
    return normalized, errors


def _collect_plan_ids(action_plan: List[Dict[str, Any]]) -> List[str]:
    ids: List[str] = []
    for action in action_plan:
        args = action.get("arguments", {})
        for key in ("targetId", "rootId", "beforeId"):
            v = args.get(key)
            if isinstance(v, str) and v.strip():
                ids.append(v.strip())
        fragment = args.get("fragment")
        if isinstance(fragment, str):
            ids.extend(_extract_ids_from_fragment(fragment))
    return ids


def _get_success_targets(blueprint: Dict[str, Any]) -> List[str]:
    success = blueprint.get("success_check", {})
    if not isinstance(success, dict):
        return []

    ids = success.get("must_have_ids", [])
    attr_items = success.get("must_have_attrs", [])
    out: List[str] = []

    if isinstance(ids, list):
        out.extend([x.strip() for x in ids if isinstance(x, str) and x.strip()])
    if isinstance(attr_items, list):
        for item in attr_items:
            if isinstance(item, dict):
                v = item.get("id")
                if isinstance(v, str) and v.strip():
                    out.append(v.strip())
    return sorted(set(out))


def _build_unified_diff(before: str, after: str, max_lines: int = 120) -> str:
    diff = list(
        difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile="before",
            tofile="after",
            lineterm="",
        )
    )
    if len(diff) > max_lines:
        half = max_lines // 2
        diff = diff[:half] + ["... <snip> ..."] + diff[-half:]
    return "\n".join(diff)


def validate_blueprint_execution(
    blueprint: Dict[str, Any],
    *,
    require_gt_action_plan: bool = True,
    max_plan_steps: int = 12,
) -> ExecutionValidationResult:
    errors: List[str] = []
    summary: Dict[str, Any] = {
        "enabled": True,
        "valid": False,
        "plan_steps": 0,
        "executed_steps": 0,
        "state_check": {},
        "plan_trace": [],
        "diff_patch": "",
    }

    if CanvasEnvironment is None:
        errors.append("CanvasEnvironment unavailable: install runtime deps (bs4, lxml, playwright)")
        return ExecutionValidationResult(valid=False, errors=errors, summary=summary)

    raw_plan = blueprint.get("optional_gt_action_plan")
    if raw_plan is None:
        if require_gt_action_plan:
            errors.append("missing optional_gt_action_plan")
            return ExecutionValidationResult(valid=False, errors=errors, summary=summary)
        raw_plan = []

    action_plan, parse_errors = _normalize_action_plan(raw_plan)
    if parse_errors:
        errors.extend(parse_errors)
        return ExecutionValidationResult(valid=False, errors=errors, summary=summary)

    summary["plan_steps"] = len(action_plan)

    if require_gt_action_plan and len(action_plan) == 0:
        errors.append("optional_gt_action_plan must be non-empty")
        return ExecutionValidationResult(valid=False, errors=errors, summary=summary)

    if len(action_plan) > max_plan_steps:
        errors.append(f"optional_gt_action_plan length {len(action_plan)} exceeds max_plan_steps={max_plan_steps}")
        return ExecutionValidationResult(valid=False, errors=errors, summary=summary)

    constraints = blueprint.get("constraints", {})
    allowed_tools: List[str] = []
    if isinstance(constraints, dict) and isinstance(constraints.get("allowed_tools"), list):
        allowed_tools = [x for x in constraints["allowed_tools"] if isinstance(x, str)]
    if not allowed_tools:
        allowed_tools = list(SUPPORTED_TOOLS)
    allow = set(allowed_tools)

    for i, action in enumerate(action_plan):
        name = action.get("name", "")
        if name not in allow:
            errors.append(f"optional_gt_action_plan[{i}] tool '{name}' not in constraints.allowed_tools")
        if name not in SUPPORTED_TOOLS:
            errors.append(f"optional_gt_action_plan[{i}] tool '{name}' unsupported")
    if errors:
        return ExecutionValidationResult(valid=False, errors=errors, summary=summary)

    init_svg = str(blueprint.get("init_svg", ""))
    env = CanvasEnvironment(initial_svg=init_svg)
    before_state = env.blackboard.state

    for i, action in enumerate(action_plan):
        step_log: Dict[str, Any] = {
            "step": i,
            "name": action.get("name", ""),
            "arguments": action.get("arguments", {}),
        }
        try:
            execution = env.execute_tool_call(action, render_path=None)
            step_log["ok"] = True
            step_log["tool_result"] = execution.get("tool_result", {})
            step_log["render_result"] = execution.get("render_result", "")
            summary["executed_steps"] += 1
        except Exception as exc:
            step_log["ok"] = False
            step_log["error"] = f"{type(exc).__name__}: {exc}"
            summary["plan_trace"].append(step_log)
            errors.append(f"execution failed at optional_gt_action_plan[{i}]: {type(exc).__name__}: {exc}")
            return ExecutionValidationResult(valid=False, errors=errors, summary=summary)
        summary["plan_trace"].append(step_log)

    after_state = env.blackboard.state
    summary["diff_patch"] = _build_unified_diff(before_state, after_state)

    success = blueprint.get("success_check", {})
    must_have_ids = success.get("must_have_ids", []) if isinstance(success, dict) else []
    must_have_attrs = success.get("must_have_attrs", []) if isinstance(success, dict) else []

    id_results: List[Dict[str, Any]] = []
    missing_ids: List[str] = []
    if isinstance(must_have_ids, list):
        for element_id in must_have_ids:
            if not isinstance(element_id, str):
                continue
            ok = _id_exists(after_state, element_id)
            id_results.append({"id": element_id, "pass": ok})
            if not ok:
                missing_ids.append(element_id)

    attr_results: List[Dict[str, Any]] = []
    missing_attrs: List[Dict[str, Any]] = []
    if isinstance(must_have_attrs, list):
        for item in must_have_attrs:
            if not isinstance(item, dict):
                continue
            element_id = str(item.get("id", ""))
            attr = str(item.get("attr", ""))
            value = str(item.get("value", ""))
            ok = _attr_equals(after_state, element_id, attr, value)
            attr_results.append({"item": item, "pass": ok})
            if not ok:
                missing_attrs.append(item)

    summary["state_check"] = {
        "missing_ids": missing_ids,
        "missing_attrs": missing_attrs,
        "must_have_ids": id_results,
        "must_have_attrs": attr_results,
    }

    if missing_ids:
        errors.append(f"execution check failed: missing must_have_ids {missing_ids}")
    if missing_attrs:
        errors.append(f"execution check failed: missing must_have_attrs {missing_attrs}")

    success_targets = _get_success_targets(blueprint)
    init_ids = set(_extract_ids_from_svg(init_svg)) | BASELINE_SAFE_IDS
    non_baseline_targets = [x for x in success_targets if x not in init_ids]
    plan_ids = set(_collect_plan_ids(action_plan))
    uncovered_targets = [x for x in non_baseline_targets if x not in plan_ids and not _id_exists(after_state, x)]
    summary["target_coverage"] = {
        "success_targets": success_targets,
        "non_baseline_targets": non_baseline_targets,
        "plan_ids": sorted(plan_ids),
        "uncovered_targets": uncovered_targets,
    }
    if uncovered_targets:
        errors.append(
            "optional_gt_action_plan does not cover success_check targets: "
            f"{uncovered_targets}"
        )

    valid = len(errors) == 0
    summary["valid"] = valid
    return ExecutionValidationResult(valid=valid, errors=errors, summary=summary)

