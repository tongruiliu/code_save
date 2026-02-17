from __future__ import annotations

import hashlib
import json
import re
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
        self.target_canvas = canvas_snapshot(init_canvas_data())
        self.gt_hash = self._data_hash(init_canvas_data())
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
        gt_data = self._replay_gt_data()
        self.target_canvas = canvas_snapshot(gt_data)
        self.gt_hash = self._data_hash(gt_data)
        obs = self.critic.reset(instruction=self.task.instruction, target_canvas=self.target_canvas)
        return EnvResetResponse(observation=obs, info=EnvInfo(task=self.task, source="critic"))

    def _data_hash(self, data: Dict[str, Any]) -> str:
        payload = json.dumps(data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _replay_gt_data(self) -> Dict[str, Any]:
        gt_data = init_canvas_data()
        for action in self.task.actions:
            if action.name in self.tools_map and action.name not in self.terminate_tools:
                self.tools_map[action.name].invoke(gt_data, **action.kwargs)
        return gt_data

    def calculate_reward(self, critic_correct: bool) -> tuple[float, RewardInfo]:
        actual_hash = self._data_hash(self.data)
        gt_hash = self.gt_hash
        r_actions = actual_hash == gt_hash

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
        current_hash = self._data_hash(self.data)
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
            "target_render": json.dumps(self.target_canvas, ensure_ascii=False, sort_keys=True),
            "rendered_render": json.dumps(rendered_canvas, ensure_ascii=False, sort_keys=True),
            "target_image": json.dumps(self.target_canvas, ensure_ascii=False, sort_keys=True),
            "rendered_image": json.dumps(rendered_canvas, ensure_ascii=False, sort_keys=True),
            "matches_target": current_hash == self.gt_hash,
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

        obs = self.critic.step(assistant_message=assistant_message, context=critic_context)
        info.source = "critic"

        critic_verdict = self._parse_critic_verdict(obs) if has_answer else None
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
