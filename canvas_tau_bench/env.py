from __future__ import annotations

import base64
import hashlib
import html
import json
import mimetypes
import os
import re
import uuid
from typing import Any, Dict, List, Optional, Type

from .tools import ALL_TOOLS, Tool, canvas_snapshot, init_canvas_data
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


CANVAS_WIKI = """# Canvas Agent Policy
You are the policy model. The critic is the user.

Rules:
- You must reason step by step about visual alignment between target render and current render.
- Multi-turn behavior is required: each non-final turn should advance only one step.
- Non-final turns must contain exactly one `<think>...</think>` block with concise reasoning for this step.
- If an operation is needed for this step, include exactly one `<tool>...</tool>` block.
- The `<tool>` block must be strict JSON: {"name":"tool_name","args":{...}}.
- Final turn must be answer-only: `<answer>\\boxed{final_answer}</answer>`.
- Use one CRUD action per turn, then wait for critic feedback.
- Prefer minimal edits that directly reduce render mismatch.
- Do not output final answer until you are confident the task is solved.
- Inside `<answer>`, output only one boxed payload, with no extra text anywhere in final turn.

Output templates:
1) Edit turn
<think>short visual reasoning and next best action</think>
<tool>{"name":"insert_element","args":{"fragment":"<div id='x'>...</div>","rootId":"root"}}</tool>

2) Final turn
<answer>\\boxed{done}</answer>

Failure handling:
- If critic reports hallucination risk or answer error, verify mismatch source before next action.
- Do not repeat the same failing tool arguments; correct them explicitly.
"""

BOXED_RE = re.compile(r"^(?:\\boxed|/boxed)\{(?P<inner>.*)\}$", re.DOTALL)
FINAL_ANSWER_ONLY_RE = re.compile(
    r"^\s*<answer>\s*(?:\\boxed|/boxed)\{(?P<inner>.*)\}\s*</answer>\s*$",
    re.DOTALL,
)
VERDICT_CORRECT_RE = re.compile(r"^\s*VERDICT\s*:\s*CORRECT\s*$", re.IGNORECASE | re.MULTILINE)
VERDICT_INCORRECT_RE = re.compile(r"^\s*VERDICT\s*:\s*INCORRECT\s*$", re.IGNORECASE | re.MULTILINE)


class CanvasCRUDEnv:
    def __init__(
        self,
        tasks: List[Task],
        user_strategy: str = "scripted",
        user_model: str = "gpt-4o-mini",
        user_provider: str = "openai",
    ) -> None:
        self.tasks = tasks
        self.task_index = 0
        self.task = tasks[0]
        self.tools_map: Dict[str, Type[Tool]] = {
            t.get_info()["function"]["name"]: t for t in ALL_TOOLS
        }
        self.tools_info = [t.get_info() for t in ALL_TOOLS]
        self.wiki = CANVAS_WIKI
        self.terminate_tools = {"finish_canvas"}
        self.critic = load_user(user_strategy, model=user_model, provider=user_provider)

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

    def reset(self, task_index: Optional[int] = None) -> EnvResetResponse:
        if task_index is not None:
            self.task_index = task_index
        self.task = self.tasks[self.task_index]
        self.data = init_canvas_data()
        self.actions = []
        self.assistant_messages = []
        self.answer_history = []
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
            # If task directly provides target canvas in our state format, allow hash-based match.
            if self._looks_like_canvas_state(task_target_canvas):
                self.gt_hash = self._data_hash(task_target_canvas)

        if task_target_url:
            self.target_image_url = task_target_url
            self.target_image_path = task_target_path
        elif task_target_path:
            resolved_path = self._resolve_image_path(task_target_path)
            if resolved_path:
                self.target_image_path = resolved_path
                self.target_image_url = self._image_path_to_data_url(resolved_path)
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
            target_image_url=self.target_image_url,
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

    def _image_path_to_data_url(self, path: str) -> str:
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
        answer_only_ok = bool(FINAL_ANSWER_ONLY_RE.fullmatch(full_message.strip())) if full_message else False

        if not answer_only_ok:
            reason = (
                "Final turn must contain only <answer>\\boxed{...}</answer> with no extra text outside <answer>."
            )
        elif not boxed_format_ok:
            reason = "Final answer format is invalid. Use only \\boxed{...} inside <answer>."
        else:
            reason = "Answer format is valid."

        return {
            "has_answer": True,
            "answer_text": answer_text,
            "boxed_format_ok": boxed_format_ok,
            "boxed_value": boxed_value,
            "answer_only_ok": answer_only_ok,
            "format_ok": bool(boxed_format_ok and answer_only_ok),
            "reason": reason,
        }

    def _parse_critic_verdict(self, text: str) -> Optional[bool]:
        content = text or ""
        if VERDICT_CORRECT_RE.search(content):
            return True
        if VERDICT_INCORRECT_RE.search(content):
            return False
        return None

    def _flatten_canvas_lines(self, node: Dict[str, Any], depth: int, lines: List[str]) -> None:
        indent = "  " * depth
        node_id = str(node.get("id", ""))
        tag = str(node.get("tag", ""))
        text_val = str(node.get("text", "")).strip()
        attrs = node.get("attrs", {}) or {}
        attrs_str = ""
        if attrs:
            attrs_parts = [f"{k}={v}" for k, v in attrs.items()]
            attrs_str = " attrs[" + ", ".join(attrs_parts) + "]"
        text_str = f" text[{text_val}]" if text_val else ""
        lines.append(f"{indent}- {node_id}<{tag}>{attrs_str}{text_str}")
        for child in node.get("children", []):
            self._flatten_canvas_lines(child, depth + 1, lines)

    def _canvas_to_svg(self, snapshot: Dict[str, Any], title: str) -> str:
        lines: List[str] = [title]
        self._flatten_canvas_lines(snapshot, depth=0, lines=lines)
        row_h = 22
        width = 1200
        height = max(180, 40 + row_h * len(lines))
        svg_lines = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            f'<rect x="0" y="0" width="{width}" height="{height}" fill="white" stroke="#dddddd" />',
        ]
        for idx, line in enumerate(lines):
            y = 28 + idx * row_h
            safe = html.escape(line)
            svg_lines.append(
                f'<text x="12" y="{y}" font-family="monospace" font-size="14" fill="#222222">{safe}</text>'
            )
        svg_lines.append("</svg>")
        return "".join(svg_lines)

    def _svg_to_data_url(self, svg_text: str) -> str:
        encoded = base64.b64encode(svg_text.encode("utf-8")).decode("utf-8")
        return f"data:image/svg+xml;base64,{encoded}"

    def _render_snapshot(self, snapshot: Dict[str, Any], prefix: str, title: str) -> tuple[str, str]:
        filename = f"{prefix}-turn-{self.turn_id:03d}.svg"
        path = os.path.join(self.render_dir, filename)
        svg_text = self._canvas_to_svg(snapshot=snapshot, title=title)
        with open(path, "w", encoding="utf-8") as f:
            f.write(svg_text)
        return path, self._svg_to_data_url(svg_text)

    def step(
        self,
        action: Action,
        assistant_message: str = "",
        parsed_assistant: Optional[Dict[str, Any]] = None,
    ) -> EnvResponse:
        self.actions.append(action)
        self.assistant_messages.append(assistant_message)

        reward = 0.0
        info = EnvInfo(task=self.task)
        action_done = False

        if action.name == RESPOND_ACTION_NAME:
            tool_obs = "No tool executed in this turn."
        elif action.name in self.tools_map:
            try:
                tool_obs = self.tools_map[action.name].invoke(self.data, **action.kwargs)
            except Exception as exc:
                tool_obs = f"Error: {exc}"
            action_done = action.name in self.terminate_tools
        else:
            tool_obs = f"Unknown action {action.name}"

        rendered_canvas = canvas_snapshot(self.data)
        self.turn_id += 1
        rendered_image_path, rendered_image_url = self._render_snapshot(
            snapshot=rendered_canvas,
            prefix="rendered",
            title=f"Rendered Canvas Turn {self.turn_id}",
        )
        current_hash = self._data_hash(self.data)
        matches_target: Optional[bool] = (current_hash == self.gt_hash) if self.gt_hash else None
        answer_text = str((parsed_assistant or {}).get("answer", "")).strip()
        has_answer = bool(answer_text)
        answer_eval: Dict[str, Any]
        if has_answer:
            self.answer_history.append(answer_text)
            answer_eval = self._evaluate_answer(
                answer_text=answer_text,
                full_message=assistant_message,
            )
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
            "assistant_action": action.to_dict(),
            "assistant_parse": parsed_assistant or {},
            "tool_result": tool_obs,
            "target_canvas": self.target_canvas,
            "rendered_canvas": rendered_canvas,
            "target_image_path": self.target_image_path,
            "target_image_url": self.target_image_url,
            "rendered_image_path": rendered_image_path,
            "rendered_image_url": rendered_image_url,
            "matches_target": matches_target,
            "action_terminated": action_done,
            "answer_evaluation": answer_eval,
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

        if has_answer and not answer_eval.get("format_ok", False):
            critic_context["answer_error_notice"] = (
                "Answer format is invalid. Final turn must be exactly <answer>\\boxed{...}</answer>."
            )
        if has_answer:
            critic_context["critic_output_format"] = (
                "If evaluating an answer, respond with:\n"
                "VERDICT: CORRECT or VERDICT: INCORRECT\n"
                "REASON: <short reason>\n"
                "FEEDBACK: <next-step feedback or closure>"
            )

        critic_feedback = self.critic.step(assistant_message=assistant_message, context=critic_context)
        info.source = "critic"

        # Canvas-style assistant feedback bundle: include render result and critique.
        assistant_feedback_payload = {
            "type": "turn_feedback_bundle",
            "tool_response": tool_obs,
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
            info.user_cost = self.critic.get_total_cost()
        elif has_answer and critic_verdict is False:
            info.user_cost = self.critic.get_total_cost()
        elif has_answer and critic_verdict is None:
            info.user_cost = self.critic.get_total_cost()

        return EnvResponse(observation=obs, reward=reward, done=done, info=info)
