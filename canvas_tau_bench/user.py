from __future__ import annotations

import abc
from fractions import Fraction
import json
import re
from typing import Any, Dict, List, Optional

from litellm import completion


class BaseUserSimulation(abc.ABC):
    @abc.abstractmethod
    def reset(
        self,
        instruction: Optional[str] = None,
        target_canvas: Optional[Dict[str, Any]] = None,
        target_image_url: Optional[str] = None,
    ) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, assistant_message: str, context: Optional[Dict[str, Any]] = None) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def get_total_cost(self) -> float:
        raise NotImplementedError


class HumanUserSimulation(BaseUserSimulation):
    def reset(
        self,
        instruction: Optional[str] = None,
        target_canvas: Optional[Dict[str, Any]] = None,
        target_image_url: Optional[str] = None,
    ) -> str:
        prompt = instruction or "Task:"
        print(prompt)
        if target_canvas is not None:
            print("Target render snapshot:")
            print(json.dumps(target_canvas, ensure_ascii=False, indent=2))
        if target_image_url:
            print(f"Target image url: {target_image_url[:120]}...")
        return input("critic> ")

    def step(self, assistant_message: str, context: Optional[Dict[str, Any]] = None) -> str:
        print("assistant:")
        print(assistant_message)
        if context is not None:
            print("context:")
            print(json.dumps(context, ensure_ascii=False, indent=2))
            if context.get("answer_check") is not None:
                print("Please reply with VERDICT: CORRECT or VERDICT: INCORRECT.")
        return input("critic> ")

    def get_total_cost(self) -> float:
        return 0.0


class ScriptedUserSimulation(BaseUserSimulation):
    def __init__(self) -> None:
        self.instruction = ""
        self.turn = 0

    def reset(
        self,
        instruction: Optional[str] = None,
        target_canvas: Optional[Dict[str, Any]] = None,
        target_image_url: Optional[str] = None,
    ) -> str:
        self.instruction = instruction or ""
        self.turn = 0
        return (
            "You are evaluated by a critic in a visual reasoning loop. "
            f"Task: {self.instruction} "
            "For non-final turns: provide <think>...</think>; use exactly one <tool>...</tool> if an edit is needed. "
            "For final turn: output only <answer>\\boxed{final_answer}</answer> with no extra text."
        )

    def _normalize_token(self, text: str) -> str:
        return re.sub(r"\s+", "", (text or "").strip().lower())

    def _parse_numeric(self, text: str) -> Optional[float]:
        t = self._normalize_token(text)
        if not t:
            return None
        m = re.match(r"^([+-]?[0-9]*\.?[0-9]+)([a-z%]+)?$", t)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return None
        if re.match(r"^[+-]?[0-9]+/[0-9]+$", t):
            try:
                return float(Fraction(t))
            except (ValueError, ZeroDivisionError):
                return None
        return None

    def _symbol_number_match(self, model_answer: str, gold_answer: str, instruction: str) -> bool:
        m = self._normalize_token(model_answer)
        g = self._normalize_token(gold_answer)
        inst = instruction.lower().replace(" ", "")
        if re.match(r"^[a-z]$", m) and re.match(r"^[+-]?[0-9]*\.?[0-9]+$", g):
            return any([
                f"{m}={g}" in inst,
                f"{m}:{g}" in inst,
                f"{g}->{m}" in inst,
                f"{g}to{m}" in inst,
            ])
        if re.match(r"^[a-z]$", g) and re.match(r"^[+-]?[0-9]*\.?[0-9]+$", m):
            return any([
                f"{g}={m}" in inst,
                f"{g}:{m}" in inst,
                f"{m}->{g}" in inst,
                f"{m}to{g}" in inst,
            ])
        return False

    def _equivalent_answer(self, model_answer: str, gold_answer: str, instruction: str) -> bool:
        m = self._normalize_token(model_answer)
        g = self._normalize_token(gold_answer)
        if not m or not g:
            return False
        if m == g:
            return True

        mn = self._parse_numeric(m)
        gn = self._parse_numeric(g)
        if mn is not None and gn is not None:
            return abs(mn - gn) <= 1e-9

        if self._symbol_number_match(model_answer, gold_answer, instruction):
            return True
        return False

    def step(self, assistant_message: str, context: Optional[Dict[str, Any]] = None) -> str:
        self.turn += 1
        context = context or {}
        answer_eval = context.get("answer_evaluation", {}) or {}
        answer_check = context.get("answer_check", {}) or {}
        has_answer = bool(answer_eval.get("has_answer"))
        answer_format_ok = bool(answer_eval.get("format_ok"))
        tool_result = str(context.get("tool_result", ""))
        matches_target_raw = context.get("matches_target", None)
        has_match_signal = isinstance(matches_target_raw, bool)
        matches_target = bool(matches_target_raw)
        action_terminated = bool(context.get("action_terminated"))

        if has_answer:
            model_answer = str(answer_check.get("model_answer_boxed", ""))
            gold_answers = [str(x) for x in answer_check.get("gold_answers", [])]
            if not answer_format_ok:
                return (
                    "VERDICT: INCORRECT\n"
                    "REASON: Final answer format is invalid. It must be answer-only with \\boxed{...}.\n"
                    "FEEDBACK: Student answer is incorrect. Check hallucination risk and output "
                    "<answer>\\boxed{final_answer}</answer> only."
                )

            if has_match_signal and not matches_target:
                return (
                    "VERDICT: INCORRECT\n"
                    "REASON: Rendered canvas does not match target render.\n"
                    "FEEDBACK: Student answer is incorrect. Check hallucination risk and fix the visual mismatch first."
                )

            if gold_answers:
                matched = any(self._equivalent_answer(model_answer, g, self.instruction) for g in gold_answers)
                if not matched:
                    return (
                        "VERDICT: INCORRECT\n"
                        "REASON: Model answer is not equivalent to the gold answer.\n"
                        "FEEDBACK: Student answer is incorrect. Check hallucination risk and correct the final answer."
                    )

            return (
                "VERDICT: CORRECT\n"
                "REASON: Final answer is equivalent to gold target and visual state is aligned.\n"
                "FEEDBACK: Task solved."
            )

        if has_match_signal and matches_target and not has_answer:
            return "Canvas now matches target. Provide answer-only final output: <answer>\\boxed{...}</answer>."
        if tool_result.startswith("Error:"):
            return "The last tool action failed. Fix the arguments and try again with one CRUD action."
        if action_terminated and not has_answer:
            return "finish_canvas was called but final answer is missing. Provide answer-only final output."
        if self.turn <= 2:
            return "Target not matched yet. Continue with one concrete CRUD operation."
        if self.turn <= 4:
            return "Still mismatched against target render. Keep editing the canvas."
        return "Continue one precise step. If solved, provide answer-only final output."

    def get_total_cost(self) -> float:
        return 0.0


class LLMUserSimulation(BaseUserSimulation):
    def __init__(self, model: str, provider: str) -> None:
        self.model = model
        self.provider = provider
        self.total_cost = 0.0
        self.instruction: str = ""
        self.target_canvas: Dict[str, Any] = {}
        self.target_image_url: str = ""

    def _build_system_prompt(self) -> str:
        return (
            "You are a strict visual-task critic in a multi-turn loop.\n"
            "Rules:\n"
            "- Treat the assistant as the policy being trained for SFT distillation.\n"
            "- Control multi-turn behavior through your feedback: one step per turn.\n"
            "- Non-final turns should contain one concise think and at most one tool call.\n"
            "- Final turn must be answer-only: <answer>\\boxed{final_answer}</answer>.\n"
            "- You receive target render and current render each turn.\n"
            "- When answer_check exists, you must judge semantic equivalence intelligently.\n"
            "- Accept mathematically equivalent answers (e.g., 0.5 == 1/2).\n"
            "- Accept unitless/unit forms when units are not required by instruction (e.g., 2 == 2m).\n"
            "- If instruction implies symbol-number mapping, accept mapped equivalents.\n"
            "- If incorrect, explicitly mention hallucination risk and where the model is wrong.\n"
            "- For answer turns, output exactly:\n"
            "  VERDICT: CORRECT or VERDICT: INCORRECT\n"
            "  REASON: <short reason>\n"
            "  FEEDBACK: <next-step guidance or closure>\n"
            "- For non-answer turns, output one short actionable feedback sentence."
        )

    def _as_multimodal_user_content(
        self,
        text_blocks: List[str],
        image_urls: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = []
        for text in text_blocks:
            t = str(text or "").strip()
            if t:
                content.append({"type": "text", "text": t})
        for url in (image_urls or []):
            if isinstance(url, str) and url:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": url},
                })
        if not content:
            content.append({"type": "text", "text": ""})
        return content

    def _message_content_to_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
            return "".join(parts).strip()
        return str(content)

    def _generate_once(self, messages: List[Dict[str, Any]]) -> str:
        res = completion(
            model=self.model,
            custom_llm_provider=self.provider,
            messages=messages,
        )
        msg = res.choices[0].message
        self.total_cost += (res._hidden_params.get("response_cost") or 0.0)
        return self._message_content_to_text(msg.content)

    def reset(
        self,
        instruction: Optional[str] = None,
        target_canvas: Optional[Dict[str, Any]] = None,
        target_image_url: Optional[str] = None,
    ) -> str:
        self.instruction = str(instruction or "")
        self.target_canvas = dict(target_canvas or {})
        self.target_image_url = str(target_image_url or "")
        return (
            "Proceed step by step. Each non-final turn should include one <think> and at most one <tool>. "
            "Final turn must be answer-only: <answer>\\boxed{final_answer}</answer>."
        )

    def step(self, assistant_message: str, context: Optional[Dict[str, Any]] = None) -> str:
        ctx = dict(context or {})
        rendered_image_url = str(ctx.get("rendered_image_url", "") or "")
        answer_check = ctx.get("answer_check")
        answer_eval = ctx.get("answer_evaluation")

        text_blocks = [
            f"Question: {self.instruction}",
            f"Assistant turn: {assistant_message}",
        ]
        if self.target_canvas:
            text_blocks.append(
                "Target canvas metadata: " + json.dumps(self.target_canvas, ensure_ascii=False)
            )
        if isinstance(answer_eval, dict) and answer_eval.get("has_answer"):
            text_blocks.append(
                "Answer evaluation turn: return exactly VERDICT/REASON/FEEDBACK and judge semantic equivalence."
            )
        if isinstance(answer_check, dict):
            text_blocks.append(
                "Gold answers: " + json.dumps(answer_check.get("gold_answers", []), ensure_ascii=False)
            )
            text_blocks.append(
                "Model boxed answer: " + str(answer_check.get("model_answer_boxed", ""))
            )

        image_urls = []
        if self.target_image_url:
            image_urls.append(self.target_image_url)
        if rendered_image_url:
            image_urls.append(rendered_image_url)

        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {
                "role": "user",
                "content": self._as_multimodal_user_content(
                    text_blocks=text_blocks,
                    image_urls=image_urls,
                ),
            },
        ]
        return self._generate_once(messages)

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
