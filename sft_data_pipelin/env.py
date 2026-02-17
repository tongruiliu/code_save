from __future__ import annotations

import hashlib
import json
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
- You must reason about visual state alignment between target render and current render.
- Every turn must contain exactly one `<think>...</think>` block with concise reasoning.
- If an operation is needed, include exactly one `<tool>...</tool>` block.
- The `<tool>` block must be strict JSON: {"name":"tool_name","args":{...}}.
- If no operation is needed, you may omit `<tool>` and provide `<answer>...</answer>`.
- Use one CRUD action per turn, then wait for critic feedback.
- Prefer minimal edits that directly reduce render mismatch.
- Final termination is answer-based: provide `<answer>...</answer>` only when the task is solved.
- The final answer must satisfy required outputs (e.g., include `done` when requested).

Output templates:
1) Edit turn
<think>short visual reasoning and next best action</think>
<tool>{"name":"insert_element","args":{"fragment":"<div id='x'>...</div>","rootId":"root"}}</tool>

2) Final turn
<think>state is aligned and all constraints are satisfied</think>
<answer>done</answer>

Failure handling:
- If critic reports hallucination risk or answer error, verify mismatch source before next action.
- Do not repeat the same failing tool arguments; correct them explicitly.
"""


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

    def calculate_reward(self) -> tuple[float, RewardInfo]:
        actual_hash = self._data_hash(self.data)
        gt_hash = self.gt_hash
        r_actions = actual_hash == gt_hash

        reward = 1.0 if r_actions else 0.0
        outputs_ok: Dict[str, bool] = {}
        r_outputs = 1.0
        final_answer = self.answer_history[-1] if self.answer_history else ""
        normalized = final_answer.lower().replace(",", "")
        for output in self.task.outputs:
            ok = output.lower() in normalized
            outputs_ok[output] = ok
            if not ok:
                r_outputs = 0.0
                reward = 0.0

        return reward, RewardInfo(
            r_actions=r_actions,
            gt_data_hash=gt_hash,
            r_outputs=r_outputs,
            outputs=outputs_ok,
        )

    def _evaluate_answer(self, answer_text: str, matches_target: bool) -> Dict[str, Any]:
        normalized = answer_text.lower().replace(",", "")
        outputs_ok: Dict[str, bool] = {}
        missing_outputs: List[str] = []
        for output in self.task.outputs:
            ok = output.lower() in normalized
            outputs_ok[output] = ok
            if not ok:
                missing_outputs.append(output)

        outputs_pass = len(missing_outputs) == 0
        is_correct = bool(matches_target and outputs_pass)
        if is_correct:
            reason = "Answer is correct."
        elif not matches_target and not outputs_pass:
            reason = (
                "Rendered canvas does not match target and answer misses required outputs: "
                + ", ".join(missing_outputs)
            )
        elif not matches_target:
            reason = "Rendered canvas does not match target."
        else:
            reason = "Answer misses required outputs: " + ", ".join(missing_outputs)

        return {
            "has_answer": True,
            "answer_text": answer_text,
            "is_correct": is_correct,
            "matches_target": matches_target,
            "outputs_ok": outputs_ok,
            "reason": reason,
        }

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
            answer_eval = self._evaluate_answer(answer_text=answer_text, matches_target=(current_hash == self.gt_hash))
        else:
            answer_eval = {
                "has_answer": False,
                "answer_text": "",
                "is_correct": False,
                "matches_target": current_hash == self.gt_hash,
                "outputs_ok": {},
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
        if has_answer and not answer_eval["is_correct"]:
            critic_context["answer_error_notice"] = (
                "Student answer is incorrect. Please check whether hallucination occurred and "
                "pinpoint exactly where the model is wrong."
            )

        done = bool(has_answer and answer_eval["is_correct"])
        if done:
            obs = "###STOP###"
            info.source = "critic"
        else:
            obs = self.critic.step(assistant_message=assistant_message, context=critic_context)
            info.source = "critic"

        if done:
            reward, reward_info = self.calculate_reward()
            info.reward_info = reward_info
            info.user_cost = self.critic.get_total_cost()

        return EnvResponse(observation=obs, reward=reward, done=done, info=info)
