from __future__ import annotations

import abc
import json
from typing import Any, Dict, List, Optional

from litellm import completion


class BaseUserSimulation(abc.ABC):
    @abc.abstractmethod
    def reset(self, instruction: Optional[str] = None, target_canvas: Optional[Dict[str, Any]] = None) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, assistant_message: str, context: Optional[Dict[str, Any]] = None) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def get_total_cost(self) -> float:
        raise NotImplementedError


class HumanUserSimulation(BaseUserSimulation):
    def reset(self, instruction: Optional[str] = None, target_canvas: Optional[Dict[str, Any]] = None) -> str:
        prompt = instruction or "Task:"
        print(prompt)
        if target_canvas is not None:
            print("Target render snapshot:")
            print(json.dumps(target_canvas, ensure_ascii=False, indent=2))
        return input("critic> ")

    def step(self, assistant_message: str, context: Optional[Dict[str, Any]] = None) -> str:
        print("assistant:")
        print(assistant_message)
        if context is not None:
            print("context:")
            print(json.dumps(context, ensure_ascii=False, indent=2))
        return input("critic> ")

    def get_total_cost(self) -> float:
        return 0.0


class ScriptedUserSimulation(BaseUserSimulation):
    def __init__(self) -> None:
        self.instruction = ""
        self.turn = 0

    def reset(self, instruction: Optional[str] = None, target_canvas: Optional[Dict[str, Any]] = None) -> str:
        self.instruction = instruction or ""
        self.turn = 0
        return (
            "You are evaluated by a critic in a visual reasoning loop. "
            f"Task: {self.instruction} "
            "Each turn: provide <think>...</think>; use exactly one <tool>...</tool> if an edit is needed; "
            "use <answer>...</answer> only when you are confident the task is fully solved."
        )

    def step(self, assistant_message: str, context: Optional[Dict[str, Any]] = None) -> str:
        self.turn += 1
        context = context or {}
        answer_eval = context.get("answer_evaluation", {}) or {}
        has_answer = bool(answer_eval.get("has_answer"))
        answer_correct = bool(answer_eval.get("is_correct"))
        answer_reason = str(answer_eval.get("reason", ""))
        tool_result = str(context.get("tool_result", ""))
        matches_target = bool(context.get("matches_target"))
        action_terminated = bool(context.get("action_terminated"))

        if has_answer and not answer_correct:
            return (
                "Student answer is incorrect. Please check whether hallucination occurred and identify exactly "
                f"where the model is wrong. Reason: {answer_reason}"
            )
        if matches_target and not has_answer:
            return (
                "Canvas now matches the target render. Do not add extra edits. "
                "Provide final <answer> with required output tokens."
            )
        if tool_result.startswith("Error:"):
            return "The last tool action failed. Fix the arguments and try again with one CRUD action."
        if action_terminated and not has_answer:
            return "finish_canvas was called but final <answer> is missing. Provide a final answer."
        if self.turn <= 2:
            return "Target not matched yet. Continue with one concrete CRUD operation."
        if self.turn <= 4:
            return "Still mismatched against target render. Keep editing the canvas."
        return "If solved, provide a final <answer>. Otherwise continue with one precise CRUD action."

    def get_total_cost(self) -> float:
        return 0.0


class LLMUserSimulation(BaseUserSimulation):
    def __init__(self, model: str, provider: str) -> None:
        self.model = model
        self.provider = provider
        self.messages: List[Dict[str, Any]] = []
        self.total_cost = 0.0

    def _build_system_prompt(self) -> str:
        return (
            "You are a strict visual-task critic in a multi-turn loop.\n"
            "Rules:\n"
            "- Treat the assistant as the policy being trained for SFT distillation.\n"
            "- The assistant should produce <think>, optional <tool>, and final <answer> when solved.\n"
            "- You receive target render and current render each turn.\n"
            "- Prefer concrete, local, non-ambiguous feedback (what is wrong, why, and next action).\n"
            "- If assistant gives an answer but it is wrong, explicitly check hallucination risk and diagnose the mistake.\n"
            "- Keep feedback short and actionable.\n"
            "- You may output ###STOP### when solved, but environment termination is answer-gated."
        )

    def reset(self, instruction: Optional[str] = None, target_canvas: Optional[Dict[str, Any]] = None) -> str:
        payload = {
            "type": "task_init",
            "instruction": instruction or "",
            "target_canvas": target_canvas or {},
        }
        self.messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        return self._next_message()

    def _next_message(self) -> str:
        res = completion(
            model=self.model,
            custom_llm_provider=self.provider,
            messages=self.messages,
        )
        msg = res.choices[0].message
        self.messages.append(msg.model_dump())
        self.total_cost += (res._hidden_params.get("response_cost") or 0.0)
        return msg.content

    def step(self, assistant_message: str, context: Optional[Dict[str, Any]] = None) -> str:
        payload = {
            "type": "turn_feedback",
            "assistant_message": assistant_message,
            "context": context or {},
        }
        self.messages.append({"role": "user", "content": json.dumps(payload, ensure_ascii=False)})
        return self._next_message()

    def get_total_cost(self) -> float:
        return self.total_cost


def load_user(strategy: str, model: str, provider: str) -> BaseUserSimulation:
    strategy = strategy.lower()
    if strategy == "human":
        return HumanUserSimulation()
    if strategy == "llm":
        return LLMUserSimulation(model=model, provider=provider)
    if strategy == "scripted":
        return ScriptedUserSimulation()
    raise ValueError(f"Unknown user strategy: {strategy}")
