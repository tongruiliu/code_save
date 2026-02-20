from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from litellm import completion

from .types import (
    Action,
    RESPOND_ACTION_FIELD_NAME,
    RESPOND_ACTION_NAME,
    SolveResult,
)


THINK_RE = re.compile(r"<think>(.*?)</think>", re.IGNORECASE | re.DOTALL)
TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.IGNORECASE | re.DOTALL)
TOOL_RE = re.compile(r"<tool>(.*?)</tool>", re.IGNORECASE | re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
FUNC_STYLE_RE = re.compile(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*)\)\s*$", re.DOTALL)
CRITIQUE_PROMPT = "<tool_response><image>This is the state of notebook. Critical Check: {critical_check}</tool_response>"


def _message_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        t = content.get("text")
        if isinstance(t, str):
            return t
        return str(content)
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                item_type = str(item.get("type", "")).strip().lower()
                if item_type in {"text", "output_text", "input_text"}:
                    parts.append(str(item.get("text", "")))
                    continue
                if isinstance(item.get("text"), str):
                    parts.append(str(item.get("text", "")))
        return "".join(parts).strip()
    return str(content)


def _image_part(url: str) -> Dict[str, Any]:
    return {
        "type": "image_url",
        "image_url": {"url": url},
    }


def _user_message(content: Any) -> Dict[str, Any]:
    return {"role": "user", "content": content}


def _path_or_url_to_data_url(value: str, cache: Dict[str, str]) -> str:
    v = str(value or "").strip()
    if not v:
        return v
    if v.startswith(("data:", "http://", "https://")):
        return v
    path = v[7:] if v.startswith("file://") else v
    if not os.path.exists(path):
        return v
    if path in cache:
        return cache[path]
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
    data_url = f"data:{mime};base64,{encoded}"
    cache[path] = data_url
    return data_url


def _materialize_messages_for_model(messages: List[Dict[str, Any]], cache: Dict[str, str]) -> List[Dict[str, Any]]:
    materialized: List[Dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            new_content: List[Any] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    image_obj = dict(item.get("image_url", {}) or {})
                    url_val = str(image_obj.get("url", "") or "").strip()
                    image_obj["url"] = _path_or_url_to_data_url(url_val, cache)
                    new_item = dict(item)
                    new_item["image_url"] = image_obj
                    new_content.append(new_item)
                else:
                    new_content.append(item)
            materialized.append({"role": msg.get("role"), "content": new_content})
        else:
            materialized.append(msg)
    return materialized


def _usage_to_dict(usage: Any) -> Dict[str, int]:
    if usage is None:
        return {}
    if isinstance(usage, dict):
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens = int(getattr(usage, "total_tokens", 0) or (prompt_tokens + completion_tokens))
    if prompt_tokens == 0 and completion_tokens == 0 and total_tokens == 0:
        return {}
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _sum_usage(items: List[Dict[str, int]]) -> Dict[str, int]:
    prompt_tokens = sum(int(x.get("prompt_tokens", 0) or 0) for x in items)
    completion_tokens = sum(int(x.get("completion_tokens", 0) or 0) for x in items)
    total_tokens = sum(int(x.get("total_tokens", 0) or 0) for x in items)
    if prompt_tokens == 0 and completion_tokens == 0 and total_tokens == 0:
        return {}
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _compact_observation_for_log(observation: str) -> Any:
    try:
        payload = json.loads(observation)
    except json.JSONDecodeError:
        return observation
    if not isinstance(payload, dict):
        return payload

    msg_type = str(payload.get("type", "")).strip()
    if msg_type == "init_bundle":
        target_image_ref = str(payload.get("target_image_path", "")).strip() or str(payload.get("target_image_url", "")).strip()
        return {
            "type": "init_bundle",
            "instruction": str(payload.get("instruction", "")),
            "critic_feedback": str(payload.get("critic_feedback", "")),
            "target_image_ref": target_image_ref,
            "has_target_image": bool(target_image_ref),
        }
    if msg_type == "turn_feedback_bundle":
        rendered_image_ref = str(payload.get("rendered_image_path", "")).strip() or str(payload.get("rendered_image_url", "")).strip()
        target_image_ref = str(payload.get("target_image_path", "")).strip() or str(payload.get("target_image_url", "")).strip()
        return {
            "type": "turn_feedback_bundle",
            "tool_response": str(payload.get("tool_response", "")),
            "critic_feedback": str(payload.get("critic_feedback", "")),
            "rendered_image_ref": rendered_image_ref,
            "target_image_ref": target_image_ref,
            "has_rendered_image": bool(rendered_image_ref),
            "has_target_image": bool(target_image_ref),
        }
    return payload


def _observation_to_user_messages(observation: str) -> List[Dict[str, Any]]:
    try:
        payload = json.loads(observation)
    except json.JSONDecodeError:
        return [_user_message(observation)]
    if not isinstance(payload, dict):
        return [_user_message(observation)]

    msg_type = str(payload.get("type", "")).strip()
    if msg_type == "init_bundle":
        instruction = str(payload.get("instruction", "")).strip()
        critic_feedback = str(payload.get("critic_feedback", "")).strip()
        text_lines: List[str] = []
        if instruction:
            text_lines.append(f"Task: {instruction}")
        if critic_feedback:
            text_lines.append(critic_feedback)
        text_block = "\n".join(text_lines).strip() or observation
        content: List[Dict[str, Any]] = [{"type": "text", "text": text_block}]
        target_image_ref = str(payload.get("target_image_path", "")).strip() or str(payload.get("target_image_url", "")).strip()
        if target_image_ref:
            content.append(_image_part(target_image_ref))
        return [_user_message(content)]

    if msg_type == "turn_feedback_bundle":
        tool_response = str(payload.get("tool_response", "")).strip()
        critic_feedback = str(payload.get("critic_feedback", "")).strip()
        user_messages: List[Dict[str, Any]] = []
        if tool_response and tool_response.startswith("Error:"):
            user_messages.append(_user_message([
                {"type": "text", "text": f"<tool_response>{tool_response}</tool_response>"},
            ]))

        rendered_image_ref = str(payload.get("rendered_image_path", "")).strip() or str(payload.get("rendered_image_url", "")).strip()
        if rendered_image_ref:
            user_messages.append(_user_message([_image_part(rendered_image_ref)]))
        if critic_feedback:
            user_messages.append(_user_message([
                {
                    "type": "text",
                    "text": CRITIQUE_PROMPT.format(critical_check=critic_feedback),
                }
            ]))
        if not user_messages:
            user_messages.append(_user_message(observation))
        return user_messages

    return [_user_message(observation)]


def _extract_first_tag(text: str, pattern: re.Pattern[str]) -> str:
    m = pattern.search(text)
    return m.group(1).strip() if m else ""


def _parse_tool_block(raw_tool: str) -> Tuple[Optional[Action], Optional[Dict[str, Any]]]:
    if not raw_tool:
        return None, None

    parsed: Optional[Dict[str, Any]] = None
    try:
        obj = json.loads(raw_tool)
        if isinstance(obj, dict):
            name = obj.get("name") or obj.get("tool") or obj.get("action")
            args = obj.get("args")
            if args is None:
                args = obj.get("arguments")
            if args is None:
                args = obj.get("kwargs") or obj.get("parameters") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            if name and isinstance(args, dict):
                parsed = {"name": str(name), "args": args}
    except json.JSONDecodeError:
        parsed = None

    if parsed is None:
        m = FUNC_STYLE_RE.match(raw_tool)
        if m:
            name = m.group(1)
            args_text = m.group(2).strip()
            args: Dict[str, Any] = {}
            if args_text:
                try:
                    decoded = json.loads(args_text)
                    if isinstance(decoded, dict):
                        args = decoded
                except json.JSONDecodeError:
                    args = {}
            parsed = {"name": name, "args": args}

    if parsed is None:
        return None, None

    return Action(name=parsed["name"], kwargs=parsed["args"]), parsed


def message_to_action(message: Dict[str, Any]) -> Tuple[Action, Dict[str, Any]]:
    content = _message_content_to_text(message.get("content")).strip()
    think = _extract_first_tag(content, THINK_RE)
    raw_tool_call = _extract_first_tag(content, TOOL_CALL_RE)
    raw_tool_legacy = _extract_first_tag(content, TOOL_RE)
    raw_tool = raw_tool_call or raw_tool_legacy
    answer = _extract_first_tag(content, ANSWER_RE)

    action, tool_obj = _parse_tool_block(raw_tool)
    if action is None:
        respond_text = answer if answer else content
        action = Action(name=RESPOND_ACTION_NAME, kwargs={RESPOND_ACTION_FIELD_NAME: respond_text})

    parsed = {
        "think": think,
        "tool": tool_obj,
        "answer": answer,
        "raw_tool_call": raw_tool_call,
        "raw_tool": raw_tool,
    }
    return action, parsed


def strip_think_for_history(text: str) -> str:
    # Follow Canvas-CoT style: keep content after </think> as the assistant trace.
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()


class ToolCallingAgent:
    def __init__(
        self,
        tools_info: List[Dict[str, Any]],
        wiki: str,
        model: str,
        provider: str,
        temperature: float = 0.0,
        api_base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        self.tools_info = tools_info
        self.wiki = wiki
        self.model = model
        self.provider = provider
        self.temperature = temperature
        self.api_base_url = (api_base_url or "").strip()
        self.api_key = (api_key or "").strip()
        self.max_tokens = max_tokens if isinstance(max_tokens, int) and max_tokens > 0 else None

    def solve(self, env: Any, task_index: Optional[int] = None, max_num_steps: int = 30) -> SolveResult:
        reset_res = env.reset(task_index=task_index)
        info = reset_res.info.to_dict()

        init_obs_messages = _observation_to_user_messages(reset_res.observation)
        # Runtime context: assistant history has <think> stripped.
        context_messages: List[Dict[str, Any]] = [{"role": "system", "content": self.wiki}]
        context_messages.extend(init_obs_messages)
        # Saved trajectory: keep assistant raw content (including <think>) for SFT targets.
        trajectory_messages: List[Dict[str, Any]] = [{"role": "system", "content": self.wiki}]
        trajectory_messages.extend(init_obs_messages)

        reward = 0.0
        total_cost = 0.0
        turns: List[Dict[str, Any]] = []
        image_data_cache: Dict[str, str] = {}

        for _ in range(max_num_steps):
            completion_kwargs: Dict[str, Any] = {}
            if self.api_base_url:
                completion_kwargs["api_base"] = self.api_base_url
            if self.api_key:
                completion_kwargs["api_key"] = self.api_key
            if self.max_tokens is not None:
                completion_kwargs["max_tokens"] = self.max_tokens

            model_messages = _materialize_messages_for_model(context_messages, image_data_cache)
            res = completion(
                model=self.model,
                custom_llm_provider=self.provider,
                messages=model_messages,
                temperature=self.temperature,
                **completion_kwargs,
            )
            next_message = res.choices[0].message.model_dump()
            assistant_cost = float(res._hidden_params.get("response_cost") or 0.0)
            total_cost += assistant_cost
            assistant_usage = _usage_to_dict(getattr(res, "usage", None))

            action, parsed = message_to_action(next_message)
            assistant_raw = _message_content_to_text(next_message.get("content") or "")
            assistant_visible = strip_think_for_history(assistant_raw)

            env_res = env.step(
                action=action,
                assistant_message=assistant_raw,
                parsed_assistant=parsed,
            )
            reward = env_res.reward
            info = {**info, **env_res.info.to_dict()}
            critic_usage = dict(getattr(env, "last_critic_usage", {}) or {})
            critic_feedback_compact = _compact_observation_for_log(env_res.observation)

            turns.append(
                {
                    "assistant_raw": assistant_raw,
                    "assistant_visible": assistant_visible,
                    "assistant_think": parsed.get("think", ""),
                    "assistant_tool": parsed.get("tool"),
                    "assistant_answer": parsed.get("answer", ""),
                    "assistant_usage": assistant_usage,
                    "assistant_cost": assistant_cost,
                    "critic_usage": critic_usage,
                    "action": action.to_dict(),
                    "critic_feedback": critic_feedback_compact,
                    "critic_feedback_raw_len": len(env_res.observation or ""),
                    "reward": env_res.reward,
                    "done": env_res.done,
                }
            )

            if env_res.done:
                # Keep terminal trajectories ending at assistant output for SFT slicing.
                context_messages.append({"role": "assistant", "content": assistant_visible})
                trajectory_messages.append({"role": "assistant", "content": assistant_raw})
                break

            context_messages.append({"role": "assistant", "content": assistant_visible})
            trajectory_messages.append({"role": "assistant", "content": assistant_raw})
            next_obs_messages = _observation_to_user_messages(env_res.observation)
            context_messages.extend(next_obs_messages)
            trajectory_messages.extend(next_obs_messages)

        return SolveResult(
            reward=reward,
            messages=trajectory_messages,
            info={
                **info,
                "turns": turns,
                "assistant_usage_total": _sum_usage([t.get("assistant_usage", {}) for t in turns]),
                "critic_usage_total": _sum_usage([t.get("critic_usage", {}) for t in turns]),
            },
            total_cost=total_cost,
        )
