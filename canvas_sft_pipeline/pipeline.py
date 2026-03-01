from __future__ import annotations

import copy
import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import litellm
from litellm import completion

from .blackboard import Blackboard
from .blackboard_tools import blackboard_tools
from .io_utils import TaskItem, equivalent_answer, path_to_data_url
from .parser import parse_response
from .prompts import (
    ANSWER_JUDGE_SYSTEM,
    CRITIQUE_PROMPT,
    CRITIQUE_SYSTEM,
    CRITIQUE_SYSTEM_WO_IMG,
    FINAL_ANSWER_RETRY_PROMPT,
    build_system_prompt,
)


@dataclass
class ModelConfig:
    model: str
    provider: str
    api_key: str
    api_base_url: str
    max_tokens: Optional[int] = None
    temperature: float = 0.0
    timeout_sec: Optional[int] = 120


@dataclass
class PipelineConfig:
    policy: ModelConfig
    critic: Optional[ModelConfig]
    max_rounds: int
    max_final_retries: int
    render_root: str
    output_dir: str
    save_trace_images: bool = True


def _usage_to_dict(usage: Any) -> Dict[str, int]:
    if usage is None:
        return {}
    if isinstance(usage, dict):
        p = int(usage.get("prompt_tokens") or 0)
        c = int(usage.get("completion_tokens") or 0)
        t = int(usage.get("total_tokens") or (p + c))
        return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": t}
    p = int(getattr(usage, "prompt_tokens", 0) or 0)
    c = int(getattr(usage, "completion_tokens", 0) or 0)
    t = int(getattr(usage, "total_tokens", 0) or (p + c))
    if p == 0 and c == 0 and t == 0:
        return {}
    return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": t}


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: List[str] = []
        for x in content:
            if not isinstance(x, dict):
                continue
            if x.get("type") == "text":
                chunks.append(str(x.get("text", "")))
        return "\n".join(chunks).strip()
    return str(content)


def _materialize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            out.append(msg)
            continue
        items: List[Dict[str, Any]] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "image_url":
                items.append(item)
                continue

            image_obj = dict(item.get("image_url", {}) or {})
            url = str(image_obj.get("url", "") or "").strip()
            resolved = path_to_data_url(url)
            if not resolved:
                continue
            image_obj["url"] = resolved
            new_item = dict(item)
            new_item["image_url"] = image_obj
            items.append(new_item)
        out.append({"role": msg.get("role"), "content": items})
    return out


def _call_completion_with_retry(kwargs: Dict[str, Any], limit: int = 5, pause: float = 5.0) -> Any:
    for attempt in range(1, limit + 1):
        try:
            return completion(**kwargs)
        except Exception:
            if attempt >= limit:
                raise
            time.sleep(pause)
    raise RuntimeError("unreachable")


def _call_model(cfg: ModelConfig, messages: List[Dict[str, Any]]) -> tuple[str, Dict[str, int]]:
    kwargs: Dict[str, Any] = {
        "model": cfg.model,
        "custom_llm_provider": cfg.provider,
        "messages": _materialize_messages(messages),
        "temperature": cfg.temperature,
    }
    if cfg.api_base_url:
        kwargs["api_base"] = cfg.api_base_url
    if cfg.api_key:
        kwargs["api_key"] = cfg.api_key
    if cfg.max_tokens is not None and cfg.max_tokens > 0:
        kwargs["max_tokens"] = cfg.max_tokens
    if cfg.timeout_sec is not None and cfg.timeout_sec > 0:
        kwargs["timeout"] = cfg.timeout_sec

    res = _call_completion_with_retry(kwargs)
    msg = res.choices[0].message
    text = _message_content_to_text(getattr(msg, "content", ""))
    return text, _usage_to_dict(getattr(res, "usage", None))


def _call_critic(
    critic_cfg: Optional[ModelConfig],
    question: str,
    original_image_path: str,
    rendered_image_path: str,
) -> str:
    if critic_cfg is None:
        return "No critic configured."

    has_original = bool(path_to_data_url(original_image_path))
    system_prompt = CRITIQUE_SYSTEM if has_original else CRITIQUE_SYSTEM_WO_IMG

    content: List[Dict[str, Any]] = [
        {"type": "text", "text": question},
    ]
    if has_original:
        content.append({"type": "image_url", "image_url": {"url": original_image_path}})
    content.append({"type": "image_url", "image_url": {"url": rendered_image_path}})

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": content},
    ]
    critique, _ = _call_model(critic_cfg, messages)
    return critique.strip()


def _json_obj_from_text(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _to_bool(v: Any, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true", "1", "yes", "y"}:
            return True
        if s in {"false", "0", "no", "n"}:
            return False
    return default


def _extract_answer_candidate(parsed: Dict[str, Any]) -> str:
    boxed = str(parsed.get("boxed", "") or "").strip()
    if boxed:
        return boxed
    answer = str(parsed.get("answer", "") or "").strip()
    if answer:
        return answer
    return str(parsed.get("content", "") or "").strip()


def _answer_field_format_ok(parsed: Dict[str, Any]) -> bool:
    # Relaxed rule: assistant can output long reasoning text.
    # As long as it contains <answer>...</answer> and a boxed answer, format is accepted.
    has_answer_tag = bool(parsed.get("has_answer_tag"))
    has_boxed = bool(str(parsed.get("boxed", "") or "").strip())
    return has_answer_tag and has_boxed


def _judge_answer_with_critic(
    critic_cfg: Optional[ModelConfig],
    task: TaskItem,
    parsed: Dict[str, Any],
    assistant_raw: str,
    rendered_image_path: str,
) -> Dict[str, Any]:
    base_format_ok = _answer_field_format_ok(parsed)
    candidate = _extract_answer_candidate(parsed)

    if critic_cfg is None:
        is_correct = any(equivalent_answer(candidate, g) for g in task.answers) if task.answers else False
        if not base_format_ok:
            feedback = "Format error: please include <answer>\\boxed{...}</answer> in your response."
        elif is_correct:
            feedback = "Accepted."
        else:
            feedback = "Incorrect. Please re-check with notebook evidence and revise."
        return {
            "format_ok": base_format_ok,
            "is_correct": bool(base_format_ok and is_correct),
            "normalized_answer": candidate,
            "feedback": feedback,
            "judge_raw": "",
        }

    ref_answers_text = json.dumps(task.answers, ensure_ascii=False)
    content: List[Dict[str, Any]] = [
        {"type": "text", "text": f"Question:\n{task.question}"},
        {"type": "text", "text": f"Reference answers:\n{ref_answers_text}"},
        {"type": "text", "text": f"Assistant final output:\n{assistant_raw}"},
    ]
    if task.image_path:
        content.append({"type": "image_url", "image_url": {"url": task.image_path}})
    if rendered_image_path:
        content.append({"type": "image_url", "image_url": {"url": rendered_image_path}})

    judge_messages = [
        {"role": "system", "content": [{"type": "text", "text": ANSWER_JUDGE_SYSTEM}]},
        {"role": "user", "content": content},
    ]
    judge_raw, _ = _call_model(critic_cfg, judge_messages)
    obj = _json_obj_from_text(judge_raw)
    if ("format_ok" not in obj) and ("is_correct" not in obj):
        # One retry with explicit JSON-only constraint to stabilize judge output.
        retry_messages = judge_messages + [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Return ONLY strict JSON with keys: "
                            "format_ok, is_correct, normalized_answer, feedback. "
                            "No extra words."
                        ),
                    }
                ],
            }
        ]
        judge_raw, _ = _call_model(critic_cfg, retry_messages)
        obj = _json_obj_from_text(judge_raw)

    has_structured_fields = ("format_ok" in obj) or ("is_correct" in obj)
    if not has_structured_fields:
        # Critic didn't follow JSON contract. Reject and ask assistant to output strict format again.
        format_ok = base_format_ok
        is_correct = False
        feedback = (
            "Answer judge output was invalid. "
            "Please keep an explicit <answer>\\boxed{...}</answer> and try again."
        )
        return {
            "format_ok": format_ok,
            "is_correct": is_correct,
            "normalized_answer": candidate,
            "feedback": feedback,
            "judge_raw": judge_raw,
        }

    judge_format_ok = _to_bool(obj.get("format_ok"), default=base_format_ok)
    format_ok = bool(base_format_ok and judge_format_ok)
    is_correct = _to_bool(obj.get("is_correct"), default=False)
    normalized = str(obj.get("normalized_answer", "") or "").strip() or candidate
    feedback = str(obj.get("feedback", "") or "").strip()

    if not format_ok:
        is_correct = False
        feedback = "Format error: please include <answer>\\boxed{...}</answer> in your response."
    elif not is_correct:
        feedback = "Incorrect. Please re-check with notebook evidence and revise."
    else:
        feedback = "Accepted."

    return {
        "format_ok": format_ok,
        "is_correct": is_correct,
        "normalized_answer": normalized,
        "feedback": feedback,
        "judge_raw": judge_raw,
    }


def _build_base_fragment(image_ref: str) -> str:
    data_url = path_to_data_url(image_ref)
    if not data_url:
        return ""
    return (
        "<figure id='base_figure'>"
        f"<img id='base_image' src='{data_url}'/>"
        "</figure>"
    )


def _reset_blackboard_to_base(blackboard: Blackboard, base_fragment: str) -> None:
    blackboard.update_state(action="clear", attrs={})
    if base_fragment:
        blackboard.update_state(
            action="insert_element",
            attrs={"fragment": base_fragment, "rootId": "root"},
        )


def _render(blackboard: Blackboard, out_path: str) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ret = blackboard.render_state(out_path)
    if str(ret).strip() != "tool execute success":
        return str(ret)
    if (not os.path.isfile(out_path)) or os.path.getsize(out_path) <= 0:
        return "render produced empty file"
    return "tool execute success"


def _tool_error(name: str, args: Any) -> Optional[str]:
    allowed = {"insert_element", "modify_element", "remove_element", "replace_element", "clear"}
    if name not in allowed:
        return f"invalid tool name: {name}"
    if not isinstance(args, dict):
        return f"invalid arguments for {name}: must be object"
    req: Dict[str, List[str]] = {
        "insert_element": ["fragment", "rootId"],
        "modify_element": ["targetId", "attrs"],
        "remove_element": ["targetId"],
        "replace_element": ["targetId", "fragment"],
        "clear": [],
    }
    miss = [k for k in req[name] if k not in args]
    if miss:
        return f"missing args for {name}: {', '.join(miss)}"
    if name == "clear" and len(args) != 0:
        return "clear arguments must be {}"
    return None


def _build_init_user_message(task: TaskItem, rendered_zero_path: str) -> Dict[str, Any]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": task.instruction}]
    if task.image_path:
        content.append({"type": "image_url", "image_url": {"url": task.image_path}})
    if rendered_zero_path:
        content.append({"type": "image_url", "image_url": {"url": rendered_zero_path}})
    return {"role": "user", "content": content}


def _build_tool_feedback_message(rendered_path: str, tool_response: str) -> Dict[str, Any]:
    content: List[Dict[str, Any]] = []
    if rendered_path:
        content.append({"type": "image_url", "image_url": {"url": rendered_path}})
    content.append({"type": "text", "text": f"<tool_response>{tool_response}</tool_response>"})
    return {"role": "user", "content": content}


def _build_answer_feedback_message(rendered_path: str, feedback: str) -> Dict[str, Any]:
    content: List[Dict[str, Any]] = []
    if rendered_path:
        content.append({"type": "image_url", "image_url": {"url": rendered_path}})
    content.append(
        {
            "type": "text",
            "text": (
                f"<answer_judge>{feedback}</answer_judge>\n"
                "Please revise. Your response must include an explicit "
                "<answer>\\boxed{...}</answer> field."
            ),
        }
    )
    return {"role": "user", "content": content}


def run_task(task: TaskItem, cfg: PipelineConfig) -> Dict[str, Any]:
    session_id = f"{task.pid}-{uuid.uuid4().hex[:8]}"
    render_dir = os.path.join(cfg.render_root, session_id)
    os.makedirs(render_dir, exist_ok=True)

    blackboard = Blackboard()
    base_fragment = _build_base_fragment(task.image_path)
    if base_fragment:
        blackboard.update_state(action="insert_element", attrs={"fragment": base_fragment, "rootId": "root"})

    turn0_path = os.path.join(render_dir, "rendered-turn-000.png")
    turn0_ret = _render(blackboard, turn0_path)
    if turn0_ret != "tool execute success":
        turn0_path = ""

    last_success_state = blackboard.state
    last_success_render = turn0_path

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": build_system_prompt(blackboard_tools)}]},
        _build_init_user_message(task, turn0_path),
    ]

    assistant_turns: List[Dict[str, Any]] = []
    final_answer = ""
    success = False
    print(f"[TaskStart] task_id={task.task_id} pid={task.pid}", flush=True)

    for turn in range(1, cfg.max_rounds + 1):
        print(f"[TurnStart] task_id={task.task_id} pid={task.pid} turn={turn}", flush=True)
        raw, usage = _call_model(cfg.policy, messages)
        parsed = parse_response(raw)
        print(
            f"[TurnPolicyDone] task_id={task.task_id} pid={task.pid} turn={turn} "
            f"tool_calls={len(parsed.get('tool_calls', []))} has_answer={bool(parsed.get('has_answer_tag')) or bool(parsed.get('boxed'))}",
            flush=True,
        )

        seed_content = str(parsed.get("seed_content", "") or "").strip()
        if parsed.get("tool_calls"):
            seed_content = seed_content.split("\\boxed")[0].strip()
        if not seed_content:
            seed_content = str(parsed.get("content", "") or "").strip()
        if not seed_content:
            seed_content = raw

        messages.append({"role": "assistant", "content": [{"type": "text", "text": seed_content}]})

        turn_item: Dict[str, Any] = {
            "turn": turn,
            "assistant_raw": raw,
            "assistant_seed_content": seed_content,
            "parsed": copy.deepcopy(parsed),
            "policy_usage": usage,
            "tool_results": [],
            "rendered_image_path": last_success_render,
            "tool_response": "",
            "critic_feedback": "",
        }

        tool_calls = list(parsed.get("tool_calls", []))
        if len(tool_calls) > 8:
            tool_calls = tool_calls[:8]
            messages.append(
                _build_tool_feedback_message(
                    last_success_render,
                    "Error: too many tool calls in one turn; truncated to first 8.",
                )
            )

        latest_tool_response = ""
        latest_render_for_turn = last_success_render
        critique = ""
        no_tool_calls = len(tool_calls) == 0
        answer_judge: Dict[str, Any] = {}
        appended_answer_feedback = False

        if not no_tool_calls:
            for i, tc in enumerate(tool_calls):
                name = str(tc.get("name", "")).strip()
                args = tc.get("arguments", {})
                err = _tool_error(name, args)
                if err:
                    result = f"Error: {err}"
                    latest_tool_response = result
                    turn_item["tool_results"].append(
                        {"index": i, "name": name, "arguments": args, "result": result}
                    )
                    messages.append(_build_tool_feedback_message(last_success_render, result))
                    continue

                state_before = blackboard.state
                render_before = last_success_render
                try:
                    blackboard.update_state(action=name, attrs=args)
                    if name == "clear":
                        _reset_blackboard_to_base(blackboard, base_fragment)

                    rendered_path = os.path.join(render_dir, f"rendered-turn-{turn:03d}-tool-{i:02d}.png")
                    ret = _render(blackboard, rendered_path)
                    if ret != "tool execute success":
                        blackboard.state = state_before
                        result = f"Error: {ret}. Reverted to previous successful state."
                        messages.append(_build_tool_feedback_message(render_before, result))
                    else:
                        result = "tool execute success"
                        last_success_state = blackboard.state
                        last_success_render = rendered_path
                        latest_render_for_turn = rendered_path
                        messages.append({"role": "user", "content": [{"type": "image_url", "image_url": {"url": rendered_path}}]})
                except Exception as exc:
                    blackboard.state = state_before
                    result = f"Error: {exc}. Reverted to previous successful state."
                    messages.append(_build_tool_feedback_message(render_before, result))

                turn_item["tool_results"].append(
                    {"index": i, "name": name, "arguments": args, "result": result}
                )
                if result.startswith("Error:"):
                    latest_tool_response = result

            # Keep state/render pointers consistent after a turn.
            if not last_success_render:
                blackboard.state = last_success_state

            if not latest_tool_response:
                latest_tool_response = "tool execute success"

            if last_success_render:
                critique = _call_critic(cfg.critic, task.question, task.image_path, last_success_render)
                messages.append(
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": CRITIQUE_PROMPT.format(critical_check=critique)}],
                    }
                )

        answer_present = bool(parsed.get("has_answer_tag")) or bool(str(parsed.get("boxed", "")).strip())
        if answer_present:
            answer_judge = _judge_answer_with_critic(
                cfg.critic,
                task,
                parsed,
                raw,
                latest_render_for_turn or last_success_render,
            )
            turn_item["answer_judge"] = answer_judge
            if answer_judge.get("format_ok") and answer_judge.get("is_correct"):
                final_answer = str(answer_judge.get("normalized_answer", "") or _extract_answer_candidate(parsed)).strip()
                success = True
                turn_item["tool_response"] = latest_tool_response
                turn_item["rendered_image_path"] = latest_render_for_turn
                turn_item["critic_feedback"] = critique
                assistant_turns.append(turn_item)
                break
            messages.append(
                _build_answer_feedback_message(
                    latest_render_for_turn or last_success_render,
                    str(answer_judge.get("feedback", "") or "Answer rejected. Please revise."),
                )
            )
            appended_answer_feedback = True

        turn_item["tool_response"] = latest_tool_response
        turn_item["rendered_image_path"] = latest_render_for_turn
        turn_item["critic_feedback"] = critique
        if answer_judge:
            turn_item["answer_judge"] = answer_judge
        assistant_turns.append(turn_item)
        if no_tool_calls and not appended_answer_feedback:
            messages.append({"role": "user", "content": [{"type": "text", "text": FINAL_ANSWER_RETRY_PROMPT}]})

    if not success and cfg.max_final_retries > 0:
        for _ in range(cfg.max_final_retries):
            raw, usage = _call_model(
                cfg.policy,
                messages + [{"role": "user", "content": [{"type": "text", "text": FINAL_ANSWER_RETRY_PROMPT}]}],
            )
            parsed = parse_response(raw)
            final_text = str(parsed.get("content", "") or "").strip() or raw
            messages.append({"role": "assistant", "content": [{"type": "text", "text": final_text}]})
            assistant_turns.append(
                {
                    "turn": len(assistant_turns) + 1,
                    "assistant_raw": raw,
                    "assistant_seed_content": final_text,
                    "parsed": copy.deepcopy(parsed),
                    "policy_usage": usage,
                    "tool_results": [],
                    "rendered_image_path": last_success_render,
                    "tool_response": "",
                    "critic_feedback": "",
                }
            )
            answer_judge = _judge_answer_with_critic(
                cfg.critic,
                task,
                parsed,
                raw,
                last_success_render,
            )
            assistant_turns[-1]["answer_judge"] = answer_judge
            if answer_judge.get("format_ok") and answer_judge.get("is_correct"):
                final_answer = str(answer_judge.get("normalized_answer", "") or _extract_answer_candidate(parsed)).strip()
                success = True
                break
            messages.append(
                _build_answer_feedback_message(
                    last_success_render,
                    str(answer_judge.get("feedback", "") or "Answer rejected. Please revise."),
                )
            )

    reward = 1.0 if success else 0.0

    sft_records: List[Dict[str, Any]] = []
    assistant_turn_idx = 0
    for i, msg in enumerate(messages):
        if msg.get("role") != "assistant":
            continue
        turn_meta = assistant_turns[assistant_turn_idx] if assistant_turn_idx < len(assistant_turns) else {}
        prefix = messages[:i]
        sft_records.append(
            {
                "task_id": task.task_id,
                "pid": task.pid,
                "reward": reward,
                "messages": copy.deepcopy(prefix),
                "assistant_target": copy.deepcopy(msg.get("content")),
                "assistant_target_text": _message_content_to_text(msg.get("content")),
                "turn_meta": turn_meta,
            }
        )
        assistant_turn_idx += 1

    return {
        "task_id": task.task_id,
        "pid": task.pid,
        "question": task.question,
        "instruction": task.instruction,
        "answers": task.answers,
        "final_answer": final_answer,
        "reward": reward,
        "session_id": session_id,
        "render_dir": render_dir,
        "messages": messages,
        "turns": assistant_turns,
        "sft_records": sft_records,
    }


def run_tasks(tasks: List[TaskItem], cfg: PipelineConfig) -> List[Dict[str, Any]]:
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.render_root, exist_ok=True)

    if hasattr(litellm, "suppress_debug_info"):
        litellm.suppress_debug_info = True
    if hasattr(litellm, "set_verbose"):
        litellm.set_verbose = False

    results: List[Dict[str, Any]] = []
    for idx, task in enumerate(tasks):
        started = time.time()
        try:
            result = run_task(task, cfg)
            status = "PASS" if result["reward"] >= 1.0 else "FAIL"
            print(
                f"{status} task_id={task.task_id} pid={task.pid} "
                f"reward={result['reward']} turns={len(result['turns'])}"
            )
        except Exception as exc:
            result = {
                "task_id": task.task_id,
                "pid": task.pid,
                "question": task.question,
                "instruction": task.instruction,
                "answers": task.answers,
                "reward": 0.0,
                "error": str(exc),
                "messages": [],
                "turns": [],
                "sft_records": [],
            }
            print(f"FAIL task_id={task.task_id} pid={task.pid} reward=0.0 error={str(exc).splitlines()[0][:240]}")
        result["elapsed_sec"] = round(time.time() - started, 3)
        result["order"] = idx
        results.append(result)
    return results
