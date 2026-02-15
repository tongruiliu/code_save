from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..blueprints.io_utils import dump_jsonl, load_jsonl
from ..canvas_env import CanvasEnvironment, TOOL_SCHEMAS
from ..image_canvas import ensure_image_based_init_svg
from .checker import evaluate_success
from .model_client import ApiChatClient, ChatClient, LocalChatClient, image_path_to_data_url
from .parser import parse_model_response
from .prompts import (
    build_answer_judge_system_prompt,
    build_answer_judge_user_text,
    build_critic_feedback_text,
    build_critic_system_prompt,
    build_critic_user_text,
    build_force_final_answer_text,
    build_initial_user_text,
    build_plan_hint_text,
    build_system_prompt,
    build_tool_feedback_text,
)


def _build_chat_client(
    *,
    backend: str,
    model: str,
    base_url: str,
    api_key: str,
    api_timeout: int,
    local_device: str,
    local_max_new_tokens: int,
    trust_remote_code: bool,
) -> ChatClient:
    if backend == "api":
        api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("Missing API key. Provide --api-key or OPENAI_API_KEY.")
        return ApiChatClient(
            model=model,
            api_key=api_key,
            base_url=base_url or None,
            timeout=api_timeout,
        )

    return LocalChatClient(
        model_path=model,
        device=local_device,
        max_new_tokens=local_max_new_tokens,
        trust_remote_code=trust_remote_code,
    )


def _build_policy_client(args: Any) -> ChatClient:
    return _build_chat_client(
        backend=args.backend,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        api_timeout=args.api_timeout,
        local_device=args.local_device,
        local_max_new_tokens=args.local_max_new_tokens,
        trust_remote_code=args.trust_remote_code,
    )


def _build_critic_client(args: Any) -> Optional[ChatClient]:
    backend = args.critic_backend or args.backend
    model = args.critic_model or args.model
    base_url = args.critic_base_url if args.critic_base_url else args.base_url
    api_key = args.critic_api_key if args.critic_api_key else args.api_key
    api_timeout = args.critic_api_timeout if args.critic_api_timeout > 0 else args.api_timeout
    local_device = args.critic_local_device or args.local_device
    local_max_new_tokens = (
        args.critic_local_max_new_tokens
        if args.critic_local_max_new_tokens > 0
        else args.local_max_new_tokens
    )
    trust_remote_code = args.critic_trust_remote_code or args.trust_remote_code

    return _build_chat_client(
        backend=backend,
        model=model,
        base_url=base_url,
        api_key=api_key,
        api_timeout=api_timeout,
        local_device=local_device,
        local_max_new_tokens=local_max_new_tokens,
        trust_remote_code=trust_remote_code,
    )


def _unwrap_blueprint(row: Dict[str, Any], fallback_task_id: str) -> Dict[str, Any]:
    if isinstance(row.get("blueprint"), dict):
        bp = dict(row["blueprint"])
    else:
        bp = dict(row)
    bp.setdefault("task_id", fallback_task_id)
    return bp


def _resolve_maybe_relative(path: str, base_dir: Path) -> str:
    p = Path(path)
    if p.is_absolute():
        return str(p)
    return str((base_dir / p).resolve())


def _get_image_paths(blueprint: Dict[str, Any], base_dir: Path) -> List[str]:
    image_paths: List[str] = []
    if isinstance(blueprint.get("image_paths"), list):
        for item in blueprint["image_paths"]:
            if isinstance(item, str) and item.strip():
                image_paths.append(_resolve_maybe_relative(item.strip(), base_dir))
    elif isinstance(blueprint.get("image_path"), str) and blueprint["image_path"].strip():
        image_paths.append(_resolve_maybe_relative(blueprint["image_path"].strip(), base_dir))
    return image_paths


def _filter_tool_schemas(allowed_tools: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    allow = set(allowed_tools)
    for schema in TOOL_SCHEMAS:
        fn = schema.get("function", {})
        name = fn.get("name", "")
        if name in allow:
            out.append(schema)
    return out


def _build_user_runtime_content(
    text: str,
    image_paths: List[str],
    client: ChatClient,
) -> Any:
    base_text = text.strip()
    valid_image_paths = [p for p in image_paths if Path(p).exists()]
    if not valid_image_paths:
        return base_text

    parts: List[Dict[str, Any]] = []
    if base_text:
        parts.append({"type": "text", "text": base_text})

    for image_path in valid_image_paths:
        url = image_path
        if client.supports_multimodal:
            try:
                url = image_path_to_data_url(image_path)
            except Exception:
                continue
        parts.append({"type": "image_url", "image_url": {"url": url}})

    if not parts:
        return base_text
    if len(parts) == 1 and parts[0].get("type") == "text":
        return base_text
    return parts


def _build_sft_content(text: str, image_paths: List[str]) -> str:
    chunks: List[str] = []
    if text.strip():
        chunks.append(text.strip())
    for image_path in image_paths:
        chunks.append(f"<image>{image_path}</image>")
    return "\n".join(chunks).strip()


def _append_user_message(
    runtime_messages: List[Dict[str, Any]],
    sft_messages: List[Dict[str, str]],
    text: str,
    image_paths: List[str],
    client: ChatClient,
) -> None:
    runtime_messages.append(
        {
            "role": "user",
            "content": _build_user_runtime_content(text=text, image_paths=image_paths, client=client),
        }
    )
    sft_messages.append({"role": "user", "content": _build_sft_content(text=text, image_paths=image_paths)})


def _sanitize_name(name: str) -> str:
    safe = re.sub(r"[^0-9a-zA-Z_.-]+", "_", name).strip("_")
    return safe or "task"


def _build_render_path(
    render_dir: Path,
    task_id: str,
    turn_id: int,
    tool_id: int,
    tool_name: str,
) -> str:
    safe_task = _sanitize_name(task_id)
    safe_tool = _sanitize_name(tool_name)
    out_dir = render_dir / safe_task
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir / f"turn_{turn_id:02d}_tool_{tool_id:02d}_{safe_tool}.png")


def _safe_snippet(text: str, limit: int = 3000) -> str:
    t = (text or "").strip()
    # Avoid flooding critic context with giant base64 image payloads.
    t = re.sub(
        r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+",
        "data:image/<omitted>;base64,<omitted>",
        t,
    )
    if len(t) <= limit:
        return t
    return t[: limit // 2] + "\n<...snip...>\n" + t[-limit // 2 :]


def _extract_first_json_object(text: str) -> Optional[Dict[str, Any]]:
    raw = (text or "").strip()
    if not raw:
        return None

    if raw.startswith("{") and raw.endswith("}"):
        try:
            payload = json.loads(raw)
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass

    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
        if isinstance(payload, dict):
            return payload
    except Exception:
        return None
    return None


def _clamp_reward_delta(x: Any) -> float:
    try:
        value = float(x)
    except Exception:
        return 0.0
    if value < -1.0:
        return -1.0
    if value > 1.0:
        return 1.0
    return value


def _normalize_gt_action_plan(raw_plan: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_plan, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in raw_plan:
        if not isinstance(item, dict):
            continue
        name = item.get("name", item.get("tool", item.get("action", "")))
        if not isinstance(name, str) or not name.strip():
            continue
        args = item.get("arguments", item.get("args", item.get("kwargs", {})))
        if not isinstance(args, dict):
            continue
        out.append({"name": name.strip(), "arguments": args})
    return out


def _build_hard_filter_report(result: Dict[str, Any]) -> Dict[str, Any]:
    evaluation = result.get("evaluation", {}) if isinstance(result, dict) else {}
    reasons: List[str] = []
    details: Dict[str, Any] = {}
    turns = result.get("turns", []) if isinstance(result, dict) else []
    if not isinstance(turns, list):
        turns = []

    num_tool_calls = 0
    num_successful_tools = 0
    for t in turns:
        if not isinstance(t, dict):
            continue
        tc = t.get("tool_calls", [])
        if isinstance(tc, list):
            num_tool_calls += len(tc)
        tr = t.get("tool_results", [])
        if isinstance(tr, list):
            for x in tr:
                if isinstance(x, dict) and bool(x.get("ok", False)):
                    num_successful_tools += 1

    critic_info = result.get("critic", {}) if isinstance(result, dict) else {}
    num_critic_calls = int(critic_info.get("num_calls", 0)) if isinstance(critic_info, dict) else 0

    if num_tool_calls <= 0:
        reasons.append("no_tool_calls")
        details["num_tool_calls"] = num_tool_calls
    if num_successful_tools <= 0:
        reasons.append("no_successful_tool_calls")
        details["num_successful_tool_calls"] = num_successful_tools
    if num_critic_calls <= 0:
        reasons.append("no_critic_calls")
        details["num_critic_calls"] = num_critic_calls

    answer_check = evaluation.get("answer_check", {}) if isinstance(evaluation, dict) else {}
    if isinstance(answer_check, dict) and not answer_check.get("pass", False):
        reason = str(answer_check.get("reason", "answer_incorrect_or_missing"))
        reasons.append(f"answer_check_failed:{reason}")
        details["answer_check"] = answer_check

    if isinstance(evaluation, dict) and not evaluation.get("boxed_pass", True):
        reasons.append("boxed_missing")

    if isinstance(evaluation, dict):
        missing_ids = [x.get("id") for x in evaluation.get("must_have_ids", []) if not x.get("pass", False)]
        if missing_ids:
            reasons.append("must_have_ids_failed")
            details["missing_ids"] = missing_ids

        missing_attrs = [x.get("item") for x in evaluation.get("must_have_attrs", []) if not x.get("pass", False)]
        if missing_attrs:
            reasons.append("must_have_attrs_failed")
            details["missing_attrs"] = missing_attrs

    passed = bool(evaluation.get("pass", False)) if isinstance(evaluation, dict) else False
    if not passed and not reasons:
        reasons.append("evaluation_pass_false")

    return {
        "pass": passed,
        "reasons": reasons,
        "details": details,
    }


def _collect_single_rollout(
    blueprint: Dict[str, Any],
    policy_client: ChatClient,
    critic_client: ChatClient,
    args: Any,
    input_base_dir: Path,
    render_dir: Path,
    answer_field: str = "",
    reference_answer: str = "",
) -> Dict[str, Any]:
    if CanvasEnvironment is None:
        raise RuntimeError(
            "CanvasEnvironment unavailable. Install runtime deps (bs4, lxml, playwright) to run rollouts."
        )

    task_id = str(blueprint.get("task_id", "unknown_task"))
    question = str(blueprint.get("question", "")).strip()
    init_svg_raw = str(blueprint.get("init_svg", ""))
    reveal_plan = blueprint.get("user_reveal_plan") or []
    if not isinstance(reveal_plan, list):
        reveal_plan = []
    reveal_plan = [str(x) for x in reveal_plan if isinstance(x, str) and x.strip()]

    constraints = blueprint.get("constraints", {})
    if not isinstance(constraints, dict):
        constraints = {}
    max_turns = int(constraints.get("max_turns", 8))
    max_turns = max(1, max_turns)
    allowed_tools = constraints.get("allowed_tools") or []
    if not isinstance(allowed_tools, list) or not allowed_tools:
        allowed_tools = [s["function"]["name"] for s in TOOL_SCHEMAS]
    allowed_tools = [str(x) for x in allowed_tools if isinstance(x, str) and x.strip()]
    gt_action_plan = _normalize_gt_action_plan(blueprint.get("optional_gt_action_plan"))

    tool_schemas = _filter_tool_schemas(allowed_tools)
    system_prompt = build_system_prompt(tool_schemas=tool_schemas, allowed_tools=allowed_tools)
    critic_system_prompt = build_critic_system_prompt()
    answer_judge_system_prompt = build_answer_judge_system_prompt()
    answer_judge_enabled = bool(getattr(args, "enable_answer_judge", True))
    answer_judge_temperature = float(getattr(args, "answer_judge_temperature", 0.0))
    answer_judge_max_tokens = int(getattr(args, "answer_judge_max_tokens", 256))
    answer_judge_client = critic_client if critic_client is not None else policy_client
    answer_judge_client_type = "critic" if critic_client is not None else "policy"
    answer_judge_calls = 0
    answer_judge_outputs: List[str] = []

    def _answer_judge(question_text: str, reference_candidates: List[str], predicted_boxed: str) -> Dict[str, Any]:
        nonlocal answer_judge_calls
        answer_judge_calls += 1
        user_text = build_answer_judge_user_text(
            question=question_text,
            reference_candidates=reference_candidates,
            predicted_boxed=predicted_boxed,
        )
        messages = [
            {"role": "system", "content": answer_judge_system_prompt},
            {"role": "user", "content": user_text},
        ]
        raw = answer_judge_client.generate(
            messages=messages,
            temperature=answer_judge_temperature,
            max_tokens=answer_judge_max_tokens,
        )
        answer_judge_outputs.append(raw)
        payload = _extract_first_json_object(raw) or {}
        equivalent = bool(payload.get("equivalent", False))
        reason = str(payload.get("reason", "")).strip()
        if not reason and raw.strip():
            reason = raw.strip()[:256]
        return {
            "equivalent": equivalent,
            "reason": reason,
            "raw": raw,
            "parsed": payload,
        }

    answer_judge_cb: Optional[Callable[[str, List[str], str], Dict[str, Any]]]
    if answer_judge_enabled:
        answer_judge_cb = _answer_judge
    else:
        answer_judge_cb = None

    image_paths = _get_image_paths(blueprint, base_dir=input_base_dir)
    base_image_path = image_paths[0] if image_paths else str(blueprint.get("image_path", "")).strip()
    init_svg = ensure_image_based_init_svg(
        init_svg=init_svg_raw,
        image_path=base_image_path,
    )
    blueprint_runtime = dict(blueprint)
    blueprint_runtime["init_svg"] = init_svg

    env = CanvasEnvironment(initial_svg=init_svg)

    runtime_messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    sft_messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    initial_text = build_initial_user_text(question)
    if reveal_plan:
        initial_text = f"{initial_text}\n\nKnown context (turn 1):\n{reveal_plan[0]}"
        reveal_idx = 1
    else:
        reveal_idx = 0
    _append_user_message(
        runtime_messages=runtime_messages,
        sft_messages=sft_messages,
        text=initial_text,
        image_paths=image_paths,
        client=policy_client,
    )

    assistant_outputs: List[str] = []
    critic_outputs: List[str] = []
    turns: List[Dict[str, Any]] = []
    finish_reason = "max_turns_reached"
    force_final_used = 0
    critic_reward_sum = 0.0
    total_tool_calls = 0
    total_successful_tools = 0
    plan_hint_idx = 0
    plan_hint_used = 0
    max_plan_hint_retries = max(0, int(getattr(args, "plan_hint_retries", 2)))

    for turn_id in range(max_turns):
        raw_output = policy_client.generate(
            messages=runtime_messages,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        parsed = parse_model_response(raw_output)
        assistant_outputs.append(raw_output)

        tool_calls = parsed.get("tool_calls", [])
        if not isinstance(tool_calls, list):
            tool_calls = []

        # Canvas-style history update: drop <think> block from assistant message
        # before feeding back into next-turn runtime context.
        assistant_visible = raw_output
        think_end_idx = assistant_visible.find("</think>")
        if think_end_idx != -1:
            assistant_visible = assistant_visible[think_end_idx + len("</think>") :].strip()
        else:
            assistant_visible = assistant_visible.strip()
        if not assistant_visible:
            assistant_visible = str(parsed.get("thought", "") or parsed.get("text", "")).strip()
        if tool_calls:
            assistant_visible = assistant_visible.split("<answer>", 1)[0].strip()
            assistant_visible = assistant_visible.split("\\boxed", 1)[0].strip()
            if not assistant_visible:
                assistant_visible = str(parsed.get("text", "")).strip()

        runtime_messages.append({"role": "assistant", "content": assistant_visible})
        sft_messages.append({"role": "assistant", "content": raw_output})

        turn_info: Dict[str, Any] = {
            "turn_id": turn_id,
            "assistant_raw": raw_output,
            "assistant_visible": assistant_visible,
            "assistant_text": parsed.get("text", ""),
            "assistant_thought": parsed.get("thought", ""),
            "has_boxed": bool(parsed.get("has_boxed", False)),
            "has_answer_tag": bool(parsed.get("has_answer_tag", False)),
            "has_final_answer": bool(parsed.get("has_final_answer", False)),
            "tool_calls": tool_calls,
            "tool_results": [],
        }

        if len(tool_calls) > args.max_tools_per_turn:
            tool_calls = tool_calls[: args.max_tools_per_turn]
            turn_info["tool_truncated"] = True

        if tool_calls:
            total_tool_calls += len(tool_calls)
            for tool_id, tool_call in enumerate(tool_calls):
                tool_name = str(tool_call.get("name", ""))
                tool_ok = False
                tool_detail = ""
                render_path = ""
                tool_result: Dict[str, Any] = {}

                if tool_name not in allowed_tools:
                    tool_detail = f"tool not allowed in this task: {tool_name}"
                else:
                    try:
                        maybe_render_path = _build_render_path(
                            render_dir=render_dir,
                            task_id=task_id,
                            turn_id=turn_id,
                            tool_id=tool_id,
                            tool_name=tool_name,
                        )
                        execution = env.execute_tool_call(tool_call, render_path=maybe_render_path)
                        tool_result = execution.get("tool_result", {})
                        render_result = execution.get("render_result")
                        render_path = maybe_render_path
                        tool_ok = render_result == "tool execute success"
                        tool_detail = str(render_result or "tool execute success")
                        if tool_ok and (not render_path or not Path(render_path).exists()):
                            tool_ok = False
                            tool_detail = "tool execute success but rendered image missing"
                    except Exception as exc:
                        tool_detail = f"{type(exc).__name__}: {exc}"

                feedback_text = build_tool_feedback_text(
                    tool_name=tool_name or "<invalid_tool>",
                    ok=tool_ok,
                    detail=tool_detail,
                )
                feedback_images = [render_path] if tool_ok and render_path and Path(render_path).exists() else []
                if feedback_text or feedback_images:
                    _append_user_message(
                        runtime_messages=runtime_messages,
                        sft_messages=sft_messages,
                        text=feedback_text,
                        image_paths=feedback_images,
                        client=policy_client,
                    )

                turn_info["tool_results"].append(
                    {
                        "tool_name": tool_name,
                        "ok": tool_ok,
                        "detail": tool_detail,
                        "arguments": tool_call.get("arguments", {}),
                        "tool_result": tool_result,
                        "render_path": render_path,
                    }
                )
                if tool_ok:
                    total_successful_tools += 1

            rendered_paths = [
                str(x.get("render_path", ""))
                for x in turn_info["tool_results"]
                if bool(x.get("ok", False))
                and isinstance(x.get("render_path", ""), str)
                and x.get("render_path", "")
                and Path(str(x.get("render_path", ""))).exists()
            ]

            if rendered_paths:
                # Canvas-style: critic is triggered only if this round has a successful rendered tool output.
                critic_canvas_path = rendered_paths[-1]
                critic_images = list(image_paths)
                critic_images.append(critic_canvas_path)

                critic_user_text = build_critic_user_text(
                    question=question,
                    assistant_raw=raw_output,
                    tool_results=turn_info["tool_results"],
                    canvas_state_snippet=_safe_snippet(env.blackboard.state),
                )
                critic_messages = [
                    {"role": "system", "content": critic_system_prompt},
                    {
                        "role": "user",
                        "content": _build_user_runtime_content(
                            text=critic_user_text,
                            image_paths=critic_images,
                            client=critic_client,
                        ),
                    },
                ]
                critic_raw = critic_client.generate(
                    messages=critic_messages,
                    temperature=args.critic_temperature,
                    max_tokens=args.critic_max_tokens,
                )
                critic_outputs.append(critic_raw)
                critic_payload = _extract_first_json_object(critic_raw) or {}
                reward_delta = _clamp_reward_delta(critic_payload.get("reward_delta", 0.0))
                critic_reward_sum += reward_delta

                critic_feedback = build_critic_feedback_text(critic_raw)
                _append_user_message(
                    runtime_messages=runtime_messages,
                    sft_messages=sft_messages,
                    text=critic_feedback,
                    image_paths=[],
                    client=policy_client,
                )
                turn_info["critic"] = {
                    "raw": critic_raw,
                    "parsed": critic_payload,
                    "reward_delta": reward_delta,
                    "images": critic_images,
                    "triggered": True,
                }
            else:
                turn_info["critic"] = {
                    "raw": "",
                    "parsed": {},
                    "reward_delta": 0.0,
                    "images": list(image_paths),
                    "triggered": False,
                    "skip_reason": "no_successful_rendered_tool_output",
                }

            if reveal_idx < len(reveal_plan):
                reveal_text = reveal_plan[reveal_idx]
                reveal_idx += 1
                _append_user_message(
                    runtime_messages=runtime_messages,
                    sft_messages=sft_messages,
                    text=reveal_text,
                    image_paths=[],
                    client=policy_client,
                )
                turn_info["user_reveal"] = reveal_text

            turns.append(turn_info)
            continue

        # APIGen-style bootstrap: if no tool call, inject one validated action hint from stage-1 plan.
        if total_successful_tools == 0 and plan_hint_used < max_plan_hint_retries and plan_hint_idx < len(gt_action_plan):
            next_action = gt_action_plan[plan_hint_idx]
            plan_hint_idx += 1
            plan_hint_used += 1
            hint_text = build_plan_hint_text(next_action)
            _append_user_message(
                runtime_messages=runtime_messages,
                sft_messages=sft_messages,
                text=hint_text,
                image_paths=[],
                client=policy_client,
            )
            turn_info["plan_hint_injected"] = True
            turn_info["plan_hint_action"] = next_action
            turns.append(turn_info)
            continue

        if turn_info["has_final_answer"] and total_successful_tools > 0:
            finish_reason = "boxed_answer"
            turns.append(turn_info)
            break

        finish_reason = "no_tool_call"
        turns.append(turn_info)
        break

    if finish_reason != "boxed_answer" and args.force_final_retries > 0:
        for _ in range(args.force_final_retries):
            force_final_used += 1
            force_text = build_force_final_answer_text()
            _append_user_message(
                runtime_messages=runtime_messages,
                sft_messages=sft_messages,
                text=force_text,
                image_paths=[],
                client=policy_client,
            )

            raw_output = policy_client.generate(
                messages=runtime_messages,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            parsed = parse_model_response(raw_output)
            assistant_outputs.append(raw_output)

            assistant_visible = raw_output.strip()
            if not assistant_visible:
                assistant_visible = str(parsed.get("text", "") or parsed.get("thought", "")).strip()
            runtime_messages.append({"role": "assistant", "content": assistant_visible})
            sft_messages.append({"role": "assistant", "content": raw_output})

            turn_info = {
                "turn_id": len(turns),
                "assistant_raw": raw_output,
                "assistant_visible": assistant_visible,
                "assistant_text": parsed.get("text", ""),
                "assistant_thought": parsed.get("thought", ""),
                "has_boxed": bool(parsed.get("has_boxed", False)),
                "has_answer_tag": bool(parsed.get("has_answer_tag", False)),
                "has_final_answer": bool(parsed.get("has_final_answer", False)),
                "tool_calls": [],
                "tool_results": [],
                "force_final_prompt": True,
            }
            turns.append(turn_info)

            if turn_info["has_final_answer"]:
                finish_reason = "boxed_answer"
                break

    final_html = env.blackboard.state
    answer_for_check = answer_field or reference_answer
    eval_result = evaluate_success(
        blueprint=blueprint_runtime,
        final_state_html=final_html,
        assistant_outputs=assistant_outputs,
        reference_answer=answer_for_check,
        question=question,
        answer_judge=answer_judge_cb,
    )
    success = bool(eval_result.get("pass", False))

    return {
        "task_id": task_id,
        "success": success,
        "status": "success" if success else "failed",
        "finish_reason": finish_reason,
        "blueprint": blueprint_runtime,
        "turns": turns,
        "sft_messages": sft_messages,
        "final_canvas_state": final_html,
        "answer": answer_for_check,
        "reference_answer": reference_answer,
        "evaluation": eval_result,
        "critic": {
            "enabled": True,
            "num_calls": len(critic_outputs),
            "reward_sum": critic_reward_sum,
            "outputs": critic_outputs,
        },
        "answer_judge": {
            "enabled": answer_judge_enabled,
            "client": answer_judge_client_type,
            "num_calls": answer_judge_calls,
            "outputs": answer_judge_outputs,
        },
        "stats": {
            "num_turns": len(turns),
            "num_assistant_messages": len(assistant_outputs),
            "num_user_messages": sum(1 for x in sft_messages if x.get("role") == "user"),
            "num_critic_messages": len(critic_outputs),
        },
    }


def run_rollout_collection(args: Any) -> int:
    random.seed(args.seed)

    in_path = Path(args.input_jsonl)
    out_path = Path(args.output_jsonl)
    rej_path = (
        Path(args.rejected_jsonl)
        if args.rejected_jsonl
        else out_path.with_name(f"{out_path.stem}_filtered_out{out_path.suffix or '.jsonl'}")
    )
    input_base_dir = in_path.parent

    render_dir = Path(args.render_dir) if args.render_dir else Path("/tmp/canvas_rollout_renders")
    render_dir.mkdir(parents=True, exist_ok=True)

    rows = load_jsonl(in_path)
    if args.limit > 0:
        rows = rows[: args.limit]

    policy_client = _build_policy_client(args)
    critic_client = _build_critic_client(args)
    accepted_rows: List[Dict[str, Any]] = []
    rejected_rows: List[Dict[str, Any]] = []

    for idx, row in enumerate(rows):
        fallback_task_id = str(row.get("task_id") or f"task_{idx:06d}")
        try:
            blueprint = _unwrap_blueprint(row, fallback_task_id=fallback_task_id)
            answer_field = str(row.get("answer", "") or blueprint.get("answer", "")).strip()
            reference_answer = str(row.get("reference_answer", "") or blueprint.get("reference_answer", "")).strip()
            result = _collect_single_rollout(
                blueprint=blueprint,
                policy_client=policy_client,
                critic_client=critic_client,
                args=args,
                input_base_dir=input_base_dir,
                render_dir=render_dir,
                answer_field=answer_field,
                reference_answer=reference_answer,
            )

            hard_filter = _build_hard_filter_report(result)
            if not hard_filter["pass"]:
                rejected_rows.append(
                    {
                        "task_id": result.get("task_id", fallback_task_id),
                        "status": "hard_filter_failed",
                        "hard_filter": hard_filter,
                        "rollout": result,
                    }
                )
                print(f"[REJECT][{fallback_task_id}] hard filter failed: {hard_filter['reasons']}")
                continue

            accepted_rows.append(result)
            print(
                f"[ACCEPT][{fallback_task_id}] "
                f"success={result['success']} turns={result['stats']['num_turns']}"
            )
        except Exception as exc:
            rejected_rows.append(
                {
                    "task_id": fallback_task_id,
                    "status": "rollout_failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "source_row": row,
                }
            )
            print(f"[REJECT][{fallback_task_id}] {type(exc).__name__}: {exc}")

    dump_jsonl(out_path, accepted_rows)
    dump_jsonl(rej_path, rejected_rows)

    print(
        json.dumps(
            {
                "input": len(rows),
                "accepted": len(accepted_rows),
                "rejected": len(rejected_rows),
                "output_jsonl": str(out_path),
                "rejected_jsonl": str(rej_path),
            },
            ensure_ascii=False,
        )
    )
    return 0
