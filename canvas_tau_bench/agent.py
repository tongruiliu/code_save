from __future__ import annotations

import json
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
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
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
        target_image_url = str(payload.get("target_image_url", "")).strip()
        if target_image_url:
            content.append(_image_part(target_image_url))
        return [_user_message(content)]

    if msg_type == "turn_feedback_bundle":
        tool_response = str(payload.get("tool_response", "")).strip()
        critic_feedback = str(payload.get("critic_feedback", "")).strip()
        user_messages: List[Dict[str, Any]] = []
        if tool_response and tool_response.startswith("Error:"):
            user_messages.append(_user_message([
                {"type": "text", "text": f"<tool_response>{tool_response}</tool_response>"},
            ]))

        rendered_image_url = str(payload.get("rendered_image_url", "")).strip()
        if rendered_image_url:
            user_messages.append(_user_message([_image_part(rendered_image_url)]))
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
    ) -> None:
        self.tools_info = tools_info
        self.wiki = wiki
        self.model = model
        self.provider = provider
        self.temperature = temperature

    def solve(self, env: Any, task_index: Optional[int] = None, max_num_steps: int = 30) -> SolveResult:
        reset_res = env.reset(task_index=task_index)
        info = reset_res.info.to_dict()

        messages: List[Dict[str, Any]] = [{"role": "system", "content": self.wiki}]
        messages.extend(_observation_to_user_messages(reset_res.observation))

        reward = 0.0
        total_cost = 0.0
        turns: List[Dict[str, Any]] = []

        for _ in range(max_num_steps):
            res = completion(
                model=self.model,
                custom_llm_provider=self.provider,
                messages=messages,
                temperature=self.temperature,
            )
            next_message = res.choices[0].message.model_dump()
            total_cost += (res._hidden_params.get("response_cost") or 0.0)

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

            turns.append(
                {
                    "assistant_raw": assistant_raw,
                    "assistant_visible": assistant_visible,
                    "assistant_think": parsed.get("think", ""),
                    "assistant_tool": parsed.get("tool"),
                    "assistant_answer": parsed.get("answer", ""),
                    "action": action.to_dict(),
                    "critic_feedback": env_res.observation,
                    "reward": env_res.reward,
                    "done": env_res.done,
                }
            )

            if env_res.done:
                # Keep terminal trajectories ending at assistant output for SFT slicing.
                messages.append({"role": "assistant", "content": assistant_visible})
                break

            messages.append({"role": "assistant", "content": assistant_visible})
            messages.extend(_observation_to_user_messages(env_res.observation))

        return SolveResult(
            reward=reward,
            messages=messages,
            info={**info, "turns": turns},
            total_cost=total_cost,
        )
