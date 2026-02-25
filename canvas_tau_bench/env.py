from __future__ import annotations

import hashlib
import html
import json
import os
import re
import subprocess
import sys
import uuid
from typing import Any, Dict, List, Optional, Type

from bs4 import BeautifulSoup, NavigableString

from .blackboard_tools import blackboard_tools
from .canvas_blackboard_backend import Blackboard
from .tools import ALL_TOOLS, Tool, init_canvas_data
from .types import (
    Action,
    EnvInfo,
    EnvResetResponse,
    EnvResponse,
    RESPOND_ACTION_NAME,
    RewardInfo,
    Task,
)
from .user import load_user

CANVAS_WIKI_TEMPLATE = """# Objective #
You are a **Visual-Reasoning Agent**, solving complex problems by synchronizing a visual Chain-of-Thought on a virtual notebook.
The primary goal is **100% accuracy**.

# Process #
# Step 1: Think Only One Step
- You should think one more step based on the current notebook/image state.
- Output one concise `<think>...</think>` block.
- Each turn should contain only a small part of reasoning.
- Use critic feedback and tool results to fix prior mistakes.

# Step 2: Tool Call
- Immediately after `<think>`, call notebook tool(s).
- Output one or more `<tool_call>...</tool_call>` blocks.
- Use incremental edits; do not jump directly to a final dump.
- After tool call, wait for tool response and critic feedback.

# Critical Rules #
- Multi-turn behavior is required.
- Non-final turns must contain exactly one `<think>` and at least one `<tool_call>`.
- Turn 1 is NOT an exception.
- Tool-call JSON must be valid and brace-balanced.
- Use strict payload:
  {"name":"tool_name","arguments":{...}}
  ({"name":"tool_name","args":{...}} is also accepted.)
- Final turn must include:
  <answer>\\boxed{final_answer}</answer>
- In final turn, tool calls are optional.

# Notebook Operation Restrictions #
- Keep structure clean, incremental, and non-overlapping.
- Keep IDs stable; new elements must have unique ids.
- Prefer minimal edits over rewriting large blocks.

# Notebook & Tools #
The notebook is an HTML container (**Width: 800px**, Height: Auto). You have the following tools.
<tools>
{provided_tools}
</tools>

For each function call, return a JSON object with function name and arguments in `<tool_call></tool_call>`:
<tool_call>
{"name":"insert_element","arguments":{"fragment":"<div id='x'>...</div>","rootId":"root"}}
</tool_call>

# Output Templates #
1) Non-final turn
<think>one concise reasoning step</think>
<tool_call>{"name":"modify_element","arguments":{"targetId":"x","attrs":{"text":"..."}}}</tool_call>
[Optional additional <tool_call>...</tool_call> blocks in same turn]

2) Final turn
<answer>\\boxed{done}</answer>
"""

BOXED_RE = re.compile(r"^(?:\\boxed|/boxed)\{(?P<inner>.*)\}$", re.DOTALL)
VERDICT_CORRECT_RE = re.compile(r"^\s*VERDICT\s*:\s*CORRECT\s*$", re.IGNORECASE | re.MULTILINE)
VERDICT_INCORRECT_RE = re.compile(r"^\s*VERDICT\s*:\s*INCORRECT\s*$", re.IGNORECASE | re.MULTILINE)
ANSWER_TAG_PRESENT_RE = re.compile(r"<answer>", re.IGNORECASE)
ANSWER_LENIENT_RE = re.compile(r"<answer>(.*?)</(?:answer|endanswer)>", re.IGNORECASE | re.DOTALL)
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>.*?</tool_call>", re.IGNORECASE | re.DOTALL)
TOOL_LEGACY_BLOCK_RE = re.compile(r"<tool>.*?</tool>", re.IGNORECASE | re.DOTALL)


class CanvasCRUDEnv:
    def __init__(
        self,
        tasks: List[Task],
        user_strategy: str = "scripted",
        user_model: str = "gpt-4o-mini",
        user_provider: str = "openai",
        user_api_base_url: Optional[str] = None,
        user_api_key: Optional[str] = None,
        user_max_tokens: Optional[int] = None,
    ) -> None:
        self.tasks = tasks
        self.task_index = 0
        self.task = tasks[0]
        self.tools_map: Dict[str, Type[Tool]] = {
            t.get_info()["function"]["name"]: t for t in ALL_TOOLS
        }
        self.tools_info = json.loads(json.dumps(blackboard_tools))
        provided_tools = json.dumps(self.tools_info, ensure_ascii=False, indent=2)
        self.wiki = CANVAS_WIKI_TEMPLATE.replace("{provided_tools}", provided_tools)
        self.canvas_backend_actions = {
            "insert_element",
            "modify_element",
            "remove_element",
            "replace_element",
            "clear",
            "clear_element",
        }
        self.blackboard = Blackboard()
        self.critic = load_user(
            user_strategy,
            model=user_model,
            provider=user_provider,
            api_base_url=user_api_base_url,
            api_key=user_api_key,
            max_tokens=user_max_tokens,
        )

        self.data = init_canvas_data()
        self.target_canvas: Dict[str, Any] = {}
        self.gt_hash: Optional[str] = None
        self.session_id = uuid.uuid4().hex[:10]
        self.render_dir = os.path.join("/tmp", "canvas_tau_bench_svg", self.session_id)
        os.makedirs(self.render_dir, exist_ok=True)
        self.turn_id = 0
        self.target_image_path = ""
        self.target_image_url = ""
        self.actions: List[Action] = []
        self.assistant_messages: List[str] = []
        self.answer_history: List[str] = []
        self.last_critic_usage: Dict[str, int] = {}

    def reset(self, task_index: Optional[int] = None) -> EnvResetResponse:
        if task_index is not None:
            self.task_index = task_index
        self.task = self.tasks[self.task_index]
        self.data = init_canvas_data()
        self.blackboard = Blackboard()
        self.actions = []
        self.assistant_messages = []
        self.answer_history = []
        self.last_critic_usage = {}
        self.turn_id = 0
        self.target_canvas = {}
        self.target_image_path = ""
        self.target_image_url = ""
        self.gt_hash = None

        task_target_url = str(getattr(self.task, "target_image_url", "") or "").strip()
        task_target_path = str(getattr(self.task, "target_image_path", "") or "").strip()
        task_target_canvas = getattr(self.task, "target_canvas", None)

        if isinstance(task_target_canvas, dict) and task_target_canvas:
            self.target_canvas = task_target_canvas
            # Canvas backend runtime state differs from legacy tree-state target hash;
            # keep critic-driven reward as the source of truth.
            self.gt_hash = None

        if task_target_url:
            self.target_image_url = task_target_url
            self.target_image_path = task_target_path
        elif task_target_path:
            resolved_path = self._resolve_image_path(task_target_path)
            if resolved_path:
                self.target_image_path = resolved_path
                self.target_image_url = ""
        elif self.target_canvas:
            self.target_image_path, self.target_image_url = self._render_snapshot(
                snapshot=self.target_canvas,
                prefix="target",
                title="Target Canvas",
            )
        else:
            # Target visual must come from task inputs in Canvas-style setup.
            self.target_canvas = {}
            self.target_image_path = ""
            self.target_image_url = ""

        critic_opening = self.critic.reset(
            instruction=self.task.instruction,
            target_canvas=self.target_canvas,
            target_image_url=self.target_image_url or self.target_image_path,
        )
        init_payload = {
            "type": "init_bundle",
            "instruction": self.task.instruction,
            "target_canvas": self.target_canvas,
            "target_image_path": self.target_image_path,
            "target_image_url": self.target_image_url,
            "critic_feedback": critic_opening,
        }
        obs = json.dumps(init_payload, ensure_ascii=False)
        return EnvResetResponse(observation=obs, info=EnvInfo(task=self.task, source="critic"))

    def _data_hash(self, data: Dict[str, Any]) -> str:
        payload = json.dumps(data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _resolve_image_path(self, path: str) -> str:
        if not path:
            return ""
        if os.path.exists(path):
            return path
        joined = os.path.join(os.getcwd(), path)
        if os.path.exists(joined):
            return joined
        return ""

    def _looks_like_canvas_state(self, data: Dict[str, Any]) -> bool:
        return all(k in data for k in ("id", "tag", "children"))

    def calculate_reward(self, critic_correct: bool) -> tuple[float, RewardInfo]:
        actual_hash = self._data_hash(self.data)
        gt_hash = self.gt_hash or ""
        r_actions = bool(self.gt_hash and actual_hash == self.gt_hash)

        reward = 1.0 if critic_correct else 0.0
        r_outputs = 1.0 if critic_correct else 0.0
        outputs_ok = {
            "critic_correct": bool(critic_correct),
            "render_match": bool(r_actions),
        }

        return reward, RewardInfo(
            r_actions=r_actions,
            gt_data_hash=gt_hash,
            r_outputs=r_outputs,
            outputs=outputs_ok,
        )

    def _evaluate_answer(self, answer_text: str, full_message: str = "") -> Dict[str, Any]:
        stripped = answer_text.strip()
        boxed_match = BOXED_RE.fullmatch(stripped)
        boxed_format_ok = boxed_match is not None
        boxed_value = boxed_match.group("inner").strip() if boxed_match else ""
        answer_only_ok = True

        if not boxed_format_ok:
            reason = "Final answer format is invalid. Use only \\boxed{...} inside <answer>."
        else:
            reason = "Answer format is valid."

        return {
            "has_answer": True,
            "answer_text": answer_text,
            "boxed_format_ok": boxed_format_ok,
            "boxed_value": boxed_value,
            "answer_only_ok": answer_only_ok,
            "format_ok": bool(boxed_format_ok),
            "reason": reason,
        }

    def _parse_critic_verdict(self, text: str) -> Optional[bool]:
        content = text or ""
        if VERDICT_CORRECT_RE.search(content):
            return True
        if VERDICT_INCORRECT_RE.search(content):
            return False
        return None

    def _html_shell(self, title: str, body_html: str) -> str:
        safe_title = html.escape(title)
        return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{safe_title}</title>
  <style>
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 24px;
      background: #eef2f7;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #222;
    }}
    .sheet {{
      width: 820px;
      min-height: 480px;
      margin: 0 auto;
      background: #fff;
      border: 1px solid #d8dee9;
      border-radius: 10px;
      padding: 16px;
      box-shadow: 0 8px 20px rgba(20, 27, 45, 0.08);
      overflow: hidden;
    }}
    .title {{
      font-size: 14px;
      color: #5a6578;
      margin-bottom: 12px;
    }}
    #root {{
      width: 100%;
      min-height: 380px;
    }}
    svg {{
      max-width: 100%;
      height: auto;
      display: block;
    }}
  </style>
</head>
<body>
  <main class="sheet">
    <div class="title">{safe_title}</div>
    {body_html}
  </main>
</body>
</html>"""

    def _snapshot_node_to_html(self, node: Dict[str, Any]) -> str:
        tag = str(node.get("tag", "div") or "div")
        node_id = str(node.get("id", "") or "")
        attrs = dict(node.get("attrs", {}) or {})
        text = str(node.get("text", "") or "")
        children = node.get("children", []) or []

        attr_parts: List[str] = []
        if node_id:
            attr_parts.append(f'id="{html.escape(node_id, quote=True)}"')
        for k, v in attrs.items():
            if k == "id":
                continue
            attr_parts.append(
                f'{html.escape(str(k), quote=True)}="{html.escape(str(v), quote=True)}"'
            )
        attrs_str = (" " + " ".join(attr_parts)) if attr_parts else ""
        children_html = "".join(self._snapshot_node_to_html(c) for c in children)
        text_html = html.escape(text)
        return f"<{tag}{attrs_str}>{text_html}{children_html}</{tag}>"

    def _node_tag_from_fragment(self, fragment: str, fallback_tag: str, node_id: str) -> Any:
        tag = None
        try:
            parsed = BeautifulSoup(fragment or "", "html.parser")
            tag = parsed.find()
        except Exception:
            tag = None
        if tag is None:
            parsed = BeautifulSoup("", "html.parser")
            tag = parsed.new_tag(fallback_tag or "div")
        tag["id"] = node_id
        return tag

    def _data_node_to_html(self, nodes: Dict[str, Dict[str, Any]], node_id: str) -> str:
        node = nodes[node_id]
        tag = self._node_tag_from_fragment(
            fragment=str(node.get("fragment", "")),
            fallback_tag=str(node.get("tag", "div")),
            node_id=str(node.get("id", node_id)),
        )

        attrs = dict(node.get("attrs", {}) or {})
        for k, v in attrs.items():
            if k == "id":
                continue
            if isinstance(v, (dict, list)):
                tag[str(k)] = json.dumps(v, ensure_ascii=False)
            else:
                tag[str(k)] = str(v)

        children_ids = list(node.get("children", []) or [])
        for child_id in children_ids:
            if child_id not in nodes:
                continue
            child_html = self._data_node_to_html(nodes, child_id)
            child_tag = BeautifulSoup(child_html, "html.parser").find()
            if child_tag is not None:
                tag.append(child_tag)

        text_val = str(node.get("text", ""))
        if text_val:
            if children_ids:
                tag.insert(0, NavigableString(text_val))
            else:
                tag.string = text_val

        return str(tag)

    def _render_html_to_png(self, html_text: str, output_path: str) -> None:
        html_path = f"{output_path}.html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_text)

        worker_path = os.path.join(os.path.dirname(__file__), "render_worker.py")
        cmd = [
            sys.executable,
            worker_path,
            "--html-path",
            html_path,
            "--output-path",
            output_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        try:
            os.remove(html_path)
        except OSError:
            pass
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(f"Render worker failed: {err}")

    def _render_snapshot(
        self,
        snapshot: Dict[str, Any],
        prefix: str,
        title: str,
        data_state: Optional[Dict[str, Any]] = None,
    ) -> tuple[str, str]:
        filename = f"{prefix}-turn-{self.turn_id:03d}.png"
        path = os.path.join(self.render_dir, filename)

        if data_state is not None and isinstance(data_state, dict):
            try:
                nodes = ((data_state.get("canvas") or {}).get("nodes") or {})
                body_html = self._data_node_to_html(nodes, "root") if "root" in nodes else "<div id='root'></div>"
            except Exception:
                body_html = self._snapshot_node_to_html(snapshot)
        else:
            body_html = self._snapshot_node_to_html(snapshot)

        html_text = self._html_shell(title=title, body_html=body_html)
        self._render_html_to_png(html_text, path)
        # Keep observation payload short: pass local image path in trajectory,
        # and convert to data URL only right before model call.
        return path, ""

    def _render_blackboard_state(self, prefix: str) -> tuple[str, str, str]:
        filename = f"{prefix}-turn-{self.turn_id:03d}.png"
        path = os.path.join(self.render_dir, filename)
        try:
            result = self.blackboard.render_state(path)
        except Exception as exc:
            return path, "", f"Error: {exc}"
        if str(result).strip() != "tool execute success":
            return path, "", f"Error: {result}"
        return path, "", ""

    def step(
        self,
        action: Optional[Action] = None,
        actions: Optional[List[Action]] = None,
        assistant_message: str = "",
        parsed_assistant: Optional[Dict[str, Any]] = None,
    ) -> EnvResponse:
        action_list: List[Action] = []
        if isinstance(actions, list):
            action_list = [a for a in actions if isinstance(a, Action)]
        elif isinstance(action, Action):
            action_list = [action]
        primary_action = action_list[0] if action_list else Action(name=RESPOND_ACTION_NAME, kwargs={})

        if action_list:
            self.actions.extend(action_list)
        else:
            self.actions.append(primary_action)
        self.assistant_messages.append(assistant_message)

        reward = 0.0
        info = EnvInfo(task=self.task)
        raw_message = str(assistant_message or "")
        parsed = parsed_assistant or {}
        answer_text = str(parsed.get("answer", "")).strip()
        has_answer = bool(answer_text)  # strict valid answer tag path: <answer>...</answer>
        has_answer_tag = bool(ANSWER_TAG_PRESENT_RE.search(raw_message))
        is_answer_attempt = bool(has_answer or has_answer_tag)

        think_blocks = THINK_BLOCK_RE.findall(raw_message)
        think_block_count = len(think_blocks)
        parsed_think = str(parsed.get("think", "") or "").strip()
        tool_blocks = TOOL_CALL_BLOCK_RE.findall(raw_message) + TOOL_LEGACY_BLOCK_RE.findall(raw_message)
        tool_block_count = len(tool_blocks)
        parsed_tools_raw = parsed.get("tools")
        if isinstance(parsed_tools_raw, list):
            parsed_tools = [x for x in parsed_tools_raw if isinstance(x, dict) and str(x.get("name", "")).strip()]
        else:
            parsed_tool = parsed.get("tool")
            parsed_tools = [parsed_tool] if isinstance(parsed_tool, dict) and str(parsed_tool.get("name", "")).strip() else []
        parsed_tool_count = len(parsed_tools)
        policy_format_error = ""
        if not is_answer_attempt:
            if think_block_count == 0:
                policy_format_error = (
                    "Non-final turn is missing <think>. "
                    "Output exactly one <think>...</think> before <tool_call>."
                )
            elif think_block_count > 1:
                policy_format_error = (
                    "Non-final turn has multiple <think> blocks. "
                    "Output exactly one <think> block."
                )
            elif not parsed_think:
                policy_format_error = (
                    "Empty <think> block. "
                    "Provide one concise step in <think> before the tool call."
                )
            elif tool_block_count == 0:
                policy_format_error = (
                    "Non-final turn is missing <tool_call>. "
                    "Output at least one CRUD <tool_call>{\"name\":\"...\",\"arguments\":{...}}</tool_call>."
                )
            else:
                if parsed_tool_count == 0:
                    policy_format_error = (
                        "Malformed <tool_call> payload. Use strict JSON with "
                        "{\"name\":\"tool_name\",\"arguments\":{...}}."
                    )
                elif parsed_tool_count < tool_block_count:
                    policy_format_error = (
                        "At least one <tool_call> payload is malformed. Use strict JSON with "
                        "{\"name\":\"tool_name\",\"arguments\":{...}}."
                    )

        format_repair_notice = (
            "Format rejected. Re-output this turn with EXACT structure:\n"
            "<think>one concise reasoning step</think>\n"
            "<tool_call>{\"name\":\"tool_name\",\"arguments\":{...}}</tool_call>\n"
            "[Optional additional <tool_call>...</tool_call> blocks in same turn]\n"
            "Rules: include exactly one <think> and at least one <tool_call>; no extra text; "
            "JSON must be valid and brace-balanced (no extra '}' )."
        )

        tool_results: List[Dict[str, Any]] = []
        if policy_format_error:
            tool_obs = f"Error: {policy_format_error}"
        elif not action_list:
            tool_obs = "No tool executed in this turn."
        else:
            for curr_action in action_list:
                if curr_action.name in self.canvas_backend_actions:
                    try:
                        self.blackboard.update_state(action=curr_action.name, attrs=dict(curr_action.kwargs or {}))
                        curr_obs = "tool execute success"
                    except Exception as exc:
                        curr_obs = f"Error: {exc}"
                else:
                    curr_obs = f"Error: Unknown action {curr_action.name}"
                tool_results.append({
                    "action": curr_action.to_dict(),
                    "tool_response": curr_obs,
                })

            error_results = [
                tr for tr in tool_results
                if str(tr.get("tool_response", "")).startswith("Error:")
            ]
            if error_results:
                tool_obs = str(error_results[0].get("tool_response", "Error: tool execution failed"))
            elif len(tool_results) == 1:
                tool_obs = str(tool_results[0].get("tool_response", ""))
            else:
                tool_obs = f"Executed {len(tool_results)} tool calls successfully."

        rendered_canvas = {
            "backend": "canvas_blackboard",
            "state_hash": hashlib.sha256(str(getattr(self.blackboard, "state", "")).encode("utf-8")).hexdigest(),
        }
        self.turn_id += 1
        rendered_image_path, rendered_image_url, render_error = self._render_blackboard_state(prefix="rendered")
        if not tool_obs.startswith("Error:") and render_error:
            tool_obs = render_error
        self.data = {
            "backend": "canvas_blackboard",
            "state_hash": rendered_canvas["state_hash"],
        }
        current_hash = self._data_hash(self.data)
        matches_target: Optional[bool] = (current_hash == self.gt_hash) if self.gt_hash else None
        lenient_match = ANSWER_LENIENT_RE.search(raw_message)
        lenient_answer_text = lenient_match.group(1).strip() if lenient_match else ""
        has_malformed_answer = bool(has_answer_tag and not has_answer)
        answer_candidate_for_feedback = bool(has_answer or has_malformed_answer)
        answer_eval: Dict[str, Any]
        if has_answer:
            self.answer_history.append(answer_text)
            answer_eval = self._evaluate_answer(
                answer_text=answer_text,
                full_message=assistant_message,
            )
        elif has_malformed_answer:
            boxed_match = BOXED_RE.fullmatch(lenient_answer_text.strip()) if lenient_answer_text else None
            answer_eval = {
                "has_answer": True,
                "answer_text": lenient_answer_text,
                "boxed_format_ok": bool(boxed_match),
                "boxed_value": boxed_match.group("inner").strip() if boxed_match else "",
                "answer_only_ok": False,
                "format_ok": False,
                "reason": "Malformed final answer tag. Use exact <answer>...</answer>.",
            }
        else:
            answer_eval = {
                "has_answer": False,
                "answer_text": "",
                "format_ok": False,
                "answer_only_ok": False,
                "reason": "No <answer> tag found in this turn.",
            }

        critic_context = {
            "instruction": self.task.instruction,
            "assistant_message": assistant_message,
            "assistant_action": primary_action.to_dict(),
            "assistant_actions": [a.to_dict() for a in action_list],
            "assistant_parse": parsed_assistant or {},
            "tool_result": tool_obs,
            "tool_results": tool_results,
            "target_canvas": self.target_canvas,
            "rendered_canvas": rendered_canvas,
            "target_image_path": self.target_image_path,
            "target_image_url": self.target_image_url,
            "rendered_image_path": rendered_image_path,
            "rendered_image_url": rendered_image_url,
            "matches_target": matches_target,
            "answer_evaluation": answer_eval,
            "policy_format_error": policy_format_error,
            "format_repair_notice": format_repair_notice if policy_format_error else "",
        }
        if has_answer:
            critic_context["answer_check"] = {
                "gold_answers": list(self.task.outputs),
                "model_answer_raw": answer_text,
                "model_answer_boxed": answer_eval.get("boxed_value", ""),
                "answer_format_ok": bool(answer_eval.get("format_ok", False)),
                "equivalence_guidance": [
                    "Treat mathematically equivalent values as correct (e.g., 0.5 == 1/2).",
                    "Treat unitless and unit forms as equivalent unless instruction requires units (e.g., 2 == 2m).",
                    "If instruction implies symbol-number mapping, allow mapped equivalents.",
                ],
            }

        if answer_candidate_for_feedback and not answer_eval.get("format_ok", False):
            critic_context["answer_error_notice"] = (
                "Answer format is invalid. Final turn must include <answer>\\boxed{...}</answer>."
            )
        if answer_candidate_for_feedback:
            critic_context["critic_output_format"] = (
                "If evaluating an answer, respond with:\n"
                "VERDICT: CORRECT or VERDICT: INCORRECT\n"
                "REASON: <short reason>\n"
                "FEEDBACK: <next-step feedback or closure>"
            )

        if policy_format_error and not answer_candidate_for_feedback:
            critic_feedback = format_repair_notice
            self.last_critic_usage = {}
        else:
            critic_feedback = self.critic.step(assistant_message=assistant_message, context=critic_context)
            self.last_critic_usage = dict(self.critic.get_last_usage() or {})
        info.source = "critic"

        # Canvas-style assistant feedback bundle: include render result and critique.
        assistant_feedback_payload = {
            "type": "turn_feedback_bundle",
            "tool_response": tool_obs,
            "tool_results": tool_results,
            "rendered_canvas": rendered_canvas,
            "target_canvas": self.target_canvas,
            "rendered_image_path": rendered_image_path,
            "rendered_image_url": rendered_image_url,
            "target_image_path": self.target_image_path,
            "target_image_url": self.target_image_url,
            "critic_feedback": critic_feedback,
        }
        obs = json.dumps(assistant_feedback_payload, ensure_ascii=False)

        critic_verdict = self._parse_critic_verdict(critic_feedback) if has_answer else None
        done = bool(has_answer and critic_verdict is True)
        if has_answer:
            critic_context["critic_verdict"] = critic_verdict

        if done:
            reward, reward_info = self.calculate_reward(critic_correct=True)
            info.reward_info = reward_info

        return EnvResponse(observation=obs, reward=reward, done=done, info=info)
