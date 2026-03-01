from __future__ import annotations

import copy
import json
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import litellm
from litellm import completion

from .blackboard_tools import blackboard_tools
from .blackboard import Blackboard
from .io_utils import TaskItem, equivalent_answer, path_to_data_url
from .parser import parse_response
from .prompts import CRITIQUE_PROMPT, CRITIQUE_SYSTEM, FINAL_ANSWER_RETRY_PROMPT, build_system_prompt


@dataclass
class ModelConfig:
    model: str
    provider: str
    api_key: str
    api_base_url: str
    max_tokens: Optional[int] = None
    temperature: float = 0.0


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


def _materialize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            out.append(msg)
            continue
        items = []
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "image_url":
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
    res = completion(**kwargs)
    msg = res.choices[0].message
    text = msg.content if isinstance(msg.content, str) else str(msg.content)
    return text, _usage_to_dict(getattr(res, "usage", None))


def _call_critic(
    critic_cfg: Optional[ModelConfig],
    question: str,
    original_image_path: str,
    rendered_image_path: str,
) -> str:
    if critic_cfg is None:
        return "No critic configured. Continue with one precise correction step if needed."

    messages = [
        {"role": "system", "content": [{"type": "text", "text": CRITIQUE_SYSTEM}]},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Question: {question}"},
                {"type": "image_url", "image_url": {"url": original_image_path}},
                {"type": "image_url", "image_url": {"url": rendered_image_path}},
            ],
        },
    ]
    critique, _ = _call_model(critic_cfg, messages)
    return critique.strip()


def _init_blackboard_with_base_image(blackboard: Blackboard, image_ref: str) -> None:
    data_url = path_to_data_url(image_ref)
    if not data_url:
        return
    fragment = (
        "<figure id='base_figure'>"
        f"<img id='base_image' src='{data_url}'/>"
        "</figure>"
    )
    blackboard.update_state(action="insert_element", attrs={"fragment": fragment, "rootId": "root"})


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
    content: List[Dict[str, Any]] = [
        {"type": "text", "text": f"Question: {task.instruction}"},
    ]
    if task.image_path:
        content.append({"type": "text", "text": "Original image:"})
        content.append({"type": "image_url", "image_url": {"url": task.image_path}})
    if rendered_zero_path:
        content.append({"type": "text", "text": "Current notebook state:"})
        content.append({"type": "image_url", "image_url": {"url": rendered_zero_path}})
    return {"role": "user", "content": content}


def _build_feedback_user_message(rendered_path: str, critique: str, tool_response: str) -> Dict[str, Any]:
    content: List[Dict[str, Any]] = []
    if rendered_path:
        content.append({"type": "image_url", "image_url": {"url": rendered_path}})
    content.append({"type": "text", "text": CRITIQUE_PROMPT.format(critical_check=critique)})
    if tool_response:
        content.append({"type": "text", "text": f"<tool_response>{tool_response}</tool_response>"})
    return {"role": "user", "content": content}


def run_task(task: TaskItem, cfg: PipelineConfig) -> Dict[str, Any]:
    session_id = f"{task.pid}-{uuid.uuid4().hex[:8]}"
    render_dir = os.path.join(cfg.render_root, session_id)
    os.makedirs(render_dir, exist_ok=True)

    blackboard = Blackboard()
    _init_blackboard_with_base_image(blackboard, task.image_path)

    turn0_path = os.path.join(render_dir, "rendered-turn-000.png")
    turn0_ret = _render(blackboard, turn0_path)
    if turn0_ret != "tool execute success":
        turn0_path = ""

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": build_system_prompt(blackboard_tools)},
        _build_init_user_message(task, turn0_path),
    ]

    assistant_turns: List[Dict[str, Any]] = []
    final_answer = ""
    success = False

    for turn in range(1, cfg.max_rounds + 1):
        raw, usage = _call_model(cfg.policy, messages)
        parsed = parse_response(raw)
        messages.append({"role": "assistant", "content": raw})

        turn_item: Dict[str, Any] = {
            "turn": turn,
            "assistant_raw": raw,
            "parsed": copy.deepcopy(parsed),
            "policy_usage": usage,
            "tool_results": [],
            "rendered_image_path": "",
            "tool_response": "",
            "critic_feedback": "",
        }

        tool_calls = parsed.get("tool_calls", [])
        tool_response = ""
        rendered_path = ""
        critique = ""

        if tool_calls:
            latest_render_ret = ""
            for i, tc in enumerate(tool_calls):
                name = str(tc.get("name", "")).strip()
                args = tc.get("arguments", {})
                err = _tool_error(name, args)
                if err:
                    result = f"Error: {err}"
                    turn_item["tool_results"].append({"index": i, "name": name, "arguments": args, "result": result})
                    tool_response = result
                    continue

                try:
                    blackboard.update_state(action=name, attrs=args)
                    rendered_path = os.path.join(render_dir, f"rendered-turn-{turn:03d}-tool-{i:02d}.png")
                    latest_render_ret = _render(blackboard, rendered_path)
                    if latest_render_ret != "tool execute success":
                        result = f"Error: {latest_render_ret}"
                    else:
                        result = "tool execute success"
                except Exception as exc:
                    result = f"Error: {exc}"

                turn_item["tool_results"].append({"index": i, "name": name, "arguments": args, "result": result})
                if result.startswith("Error:"):
                    tool_response = result

            if not tool_response:
                if turn_item["tool_results"]:
                    if all(str(x.get("result", "")).startswith("Error:") for x in turn_item["tool_results"]):
                        tool_response = str(turn_item["tool_results"][0].get("result", "Error: tool failed"))
                    else:
                        tool_response = "tool execute success"

            if rendered_path and os.path.exists(rendered_path):
                critique = _call_critic(cfg.critic, task.question, task.image_path, rendered_path)
                messages.append(_build_feedback_user_message(rendered_path, critique, tool_response))
        
        if parsed.get("boxed"):
            final_answer = str(parsed.get("boxed", "")).strip()
            if task.answers:
                success = any(equivalent_answer(final_answer, g) for g in task.answers)
            else:
                success = True
            turn_item["tool_response"] = tool_response
            turn_item["rendered_image_path"] = rendered_path
            turn_item["critic_feedback"] = critique
            assistant_turns.append(turn_item)
            break

        turn_item["tool_response"] = tool_response
        turn_item["rendered_image_path"] = rendered_path
        turn_item["critic_feedback"] = critique
        assistant_turns.append(turn_item)

        if not tool_calls:
            messages.append({"role": "user", "content": [{"type": "text", "text": FINAL_ANSWER_RETRY_PROMPT}]})

    if not success and cfg.max_final_retries > 0:
        for _ in range(cfg.max_final_retries):
            raw, usage = _call_model(
                cfg.policy,
                messages + [{"role": "user", "content": [{"type": "text", "text": FINAL_ANSWER_RETRY_PROMPT}]}],
            )
            parsed = parse_response(raw)
            messages.append({"role": "assistant", "content": raw})
            assistant_turns.append(
                {
                    "turn": len(assistant_turns) + 1,
                    "assistant_raw": raw,
                    "parsed": copy.deepcopy(parsed),
                    "policy_usage": usage,
                    "tool_results": [],
                    "rendered_image_path": "",
                    "tool_response": "",
                    "critic_feedback": "",
                }
            )
            if parsed.get("boxed"):
                final_answer = str(parsed.get("boxed", "")).strip()
                if task.answers:
                    success = any(equivalent_answer(final_answer, g) for g in task.answers)
                else:
                    success = True
                break

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
                "assistant_target": str(msg.get("content", "")),
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
            print(f"{status} task_id={task.task_id} pid={task.pid} reward={result['reward']} turns={len(result['turns'])}")
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
