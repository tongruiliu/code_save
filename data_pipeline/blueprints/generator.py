from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from ..canvas_env import SUPPORTED_TOOLS, TOOL_SCHEMAS
from ..image_canvas import ensure_image_based_init_svg
from ..model_backends import ModelBackend

from .execution import validate_blueprint_execution
from .prompts import (
    GENERATOR_USER_TEMPLATE,
    build_generator_system_prompt,
    build_repair_user_prompt,
)
from .schema import extract_json_block, validate_blueprint


def seed_to_prompt(seed: Dict[str, Any], fallback_task_id: str) -> str:
    payload = dict(seed)
    payload.setdefault("task_id", fallback_task_id)
    payload.setdefault("question", seed.get("instruction", ""))
    payload.setdefault("image_path", seed.get("image_path", ""))
    payload.setdefault("init_svg", seed.get("init_svg", ""))
    payload.setdefault("answer", seed.get("answer", ""))
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _enforce_image_based_blueprint(bp: Dict[str, Any], seed_item: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(bp)
    seed_image_path = str(seed_item.get("image_path", "")).strip()
    if seed_image_path:
        out["image_path"] = seed_image_path
    image_path = str(out.get("image_path", "")).strip()
    out["init_svg"] = ensure_image_based_init_svg(
        init_svg=str(out.get("init_svg", "")),
        image_path=image_path,
    )
    return out


def generate_single_blueprint(
    backend: ModelBackend,
    seed_item: Dict[str, Any],
    fallback_task_id: str,
    max_retries: int,
    temperature: float,
    enable_exec_check: bool = True,
    require_gt_action_plan: bool = True,
    max_plan_steps: int = 12,
) -> Tuple[Optional[Dict[str, Any]], List[str], str, Dict[str, Any]]:
    seed_json = seed_to_prompt(seed_item, fallback_task_id)
    user_prompt = GENERATOR_USER_TEMPLATE.format(seed_json=seed_json)
    system_prompt = build_generator_system_prompt(
        tool_schemas=TOOL_SCHEMAS,
        supported_tools=SUPPORTED_TOOLS,
    )

    last_raw = ""
    last_exec_summary: Dict[str, Any] = {"enabled": enable_exec_check, "valid": False}
    for step in range(max_retries + 1):
        raw = backend.generate(system_prompt, user_prompt, temperature=temperature)
        last_raw = raw
        try:
            bp = json.loads(extract_json_block(raw))
        except Exception as exc:
            errors = [f"json parse error: {exc}"]
            if step < max_retries:
                user_prompt = build_repair_user_prompt(
                    errors=errors,
                    seed_json=seed_json,
                    supported_tools=SUPPORTED_TOOLS,
                )
                continue
            return None, errors, last_raw, last_exec_summary

        bp = _enforce_image_based_blueprint(bp, seed_item)
        val = validate_blueprint(bp)
        if val.valid and enable_exec_check:
            exec_val = validate_blueprint_execution(
                bp,
                require_gt_action_plan=require_gt_action_plan,
                max_plan_steps=max_plan_steps,
            )
            last_exec_summary = exec_val.summary
            if exec_val.valid:
                return bp, [], last_raw, last_exec_summary
            if step < max_retries:
                user_prompt = build_repair_user_prompt(
                    errors=exec_val.errors,
                    seed_json=seed_json,
                    supported_tools=SUPPORTED_TOOLS,
                )
                continue
            return None, exec_val.errors, last_raw, last_exec_summary

        if val.valid:
            last_exec_summary = {"enabled": False, "valid": False, "skipped_reason": "exec_check_disabled"}
            return bp, [], last_raw, last_exec_summary

        if step < max_retries:
            user_prompt = build_repair_user_prompt(
                errors=val.errors,
                seed_json=seed_json,
                supported_tools=SUPPORTED_TOOLS,
            )
            continue
        return None, val.errors, last_raw, last_exec_summary

    return None, ["unknown generation failure"], last_raw, last_exec_summary
