from __future__ import annotations

import abc
import base64
from fractions import Fraction
import json
import mimetypes
import os
import re
from typing import Any, Dict, List, Optional

from litellm import completion


VERDICT_CORRECT_RE = re.compile(r"^\s*VERDICT\s*:\s*CORRECT\s*$", re.IGNORECASE | re.MULTILINE)
VERDICT_INCORRECT_RE = re.compile(r"^\s*VERDICT\s*:\s*INCORRECT\s*$", re.IGNORECASE | re.MULTILINE)
REASON_LINE_RE = re.compile(r"^\s*REASON\s*:\s*(.*)$", re.IGNORECASE | re.MULTILINE)
FEEDBACK_LINE_RE = re.compile(r"^\s*FEEDBACK\s*:\s*(.*)$", re.IGNORECASE | re.MULTILINE)


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

    @abc.abstractmethod
    def get_last_usage(self) -> Dict[str, int]:
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

    def get_last_usage(self) -> Dict[str, int]:
        return {}


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
            "For non-final turns: provide <think>...</think> and exactly one <tool_call>...</tool_call>. "
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
        policy_format_error = str(context.get("policy_format_error", "") or "").strip()

        if policy_format_error and not has_answer:
            return (
                "Policy format violation: non-final turn must contain exactly one valid "
                "<tool_call>{\"name\":\"...\",\"arguments\":{...}}</tool_call>. Retry one CRUD action."
            )

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

    def get_last_usage(self) -> Dict[str, int]:
        return {}


class LLMUserSimulation(BaseUserSimulation):
    def __init__(
        self,
        model: str,
        provider: str,
        api_base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        self.model = model
        self.provider = provider
        self.api_base_url = (api_base_url or "").strip()
        self.api_key = (api_key or "").strip()
        self.max_tokens = max_tokens if isinstance(max_tokens, int) and max_tokens > 0 else None
        self.total_cost = 0.0
        self.last_usage: Dict[str, int] = {}
        self._image_data_cache: Dict[str, str] = {}
        self.instruction: str = ""
        self.target_canvas: Dict[str, Any] = {}
        self.target_image_url: str = ""

    def _build_system_prompt(self) -> str:
        return (
            "You are a strict critic/auditor. You are NOT a tutor.\n"
            "Hard bans:\n"
            "- Do NOT solve the problem.\n"
            "- Do NOT derive equations.\n"
            "- Do NOT provide calculations or final numeric/letter answers.\n"
            "- Do NOT explain full reasoning.\n"
            "Role:\n"
            "- Only audit hallucination/mismatch and policy-format compliance.\n"
            "- Give one-step corrective guidance only.\n"
            "Output policy:\n"
            "- Non-answer turns: output concise audit feedback in 1-3 sentences.\n"
            "- Answer turns: output exactly three lines:\n"
            "  VERDICT: CORRECT or VERDICT: INCORRECT\n"
            "  REASON: <short audit reason>\n"
            "  FEEDBACK: <single next action or closure>\n"
            "- No markdown list, no JSON, no code block, no extra lines."
        )

    def _sanitize_non_answer_feedback(self, text: str) -> str:
        compact = " ".join(str(text or "").replace("\n", " ").split()).strip()
        if not compact:
            return "Audit only: identify one mismatch and request one concrete next CRUD step."
        upper = compact.upper()
        if "VERDICT:" in upper:
            # LLM drifted to answer-turn format; keep non-answer channel short and actionable.
            compact = "Audit only: identify one mismatch and request one concrete next CRUD step."
        banned_markers = [
            "the answer is",
            "let's solve",
            "first,",
            "step 1",
            "equation",
            "kinetic energy",
            "potential energy",
        ]
        low = compact.lower()
        if any(x in low for x in banned_markers):
            compact = "Audit only: check hallucination/mismatch and request one precise corrective action."
        # Keep 1-3 concise sentences, avoid long teaching-style feedback.
        parts = re.split(r"(?<=[\.\!\?])\s+", compact)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) > 3:
            parts = parts[:3]
        compact = " ".join(parts).strip()
        if len(compact) > 360:
            compact = compact[:360].rsplit(" ", 1)[0].strip()
        return compact

    def _sanitize_answer_feedback(self, text: str) -> str:
        raw = str(text or "")
        verdict = None
        if VERDICT_CORRECT_RE.search(raw):
            verdict = "CORRECT"
        elif VERDICT_INCORRECT_RE.search(raw):
            verdict = "INCORRECT"
        else:
            verdict = "INCORRECT"

        reason_match = REASON_LINE_RE.search(raw)
        feedback_match = FEEDBACK_LINE_RE.search(raw)
        reason = (reason_match.group(1).strip() if reason_match else "").strip()
        feedback = (feedback_match.group(1).strip() if feedback_match else "").strip()

        if not reason:
            reason = "Audit result could not confirm required answer constraints."
        if not feedback:
            feedback = "Provide only compliant output in the required format."

        reason = " ".join(reason.split())
        feedback = " ".join(feedback.split())
        if len(reason) > 140:
            reason = reason[:140].rsplit(" ", 1)[0].strip()
        if len(feedback) > 140:
            feedback = feedback[:140].rsplit(" ", 1)[0].strip()

        return f"VERDICT: {verdict}\nREASON: {reason}\nFEEDBACK: {feedback}"

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

    def _generate_once(self, messages: List[Dict[str, Any]]) -> str:
        completion_kwargs: Dict[str, Any] = {}
        if self.api_base_url:
            completion_kwargs["api_base"] = self.api_base_url
        if self.api_key:
            completion_kwargs["api_key"] = self.api_key
        if self.max_tokens is not None:
            completion_kwargs["max_tokens"] = self.max_tokens
        model_messages = _materialize_messages_for_model(messages, self._image_data_cache)
        res = completion(
            model=self.model,
            custom_llm_provider=self.provider,
            messages=model_messages,
            **completion_kwargs,
        )
        msg = res.choices[0].message
        self.total_cost += (res._hidden_params.get("response_cost") or 0.0)
        self.last_usage = _usage_to_dict(getattr(res, "usage", None))
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
        self.last_usage = {}
        self._image_data_cache = {}
        return (
            "Proceed step by step. Each non-final turn must include one <think> and exactly one <tool_call>. "
            "Final turn must be answer-only: <answer>\\boxed{final_answer}</answer>."
        )

    def step(self, assistant_message: str, context: Optional[Dict[str, Any]] = None) -> str:
        ctx = dict(context or {})
        rendered_image_url = str(ctx.get("rendered_image_url", "") or "")
        rendered_image_path = str(ctx.get("rendered_image_path", "") or "")
        target_image_path = str(ctx.get("target_image_path", "") or "")
        answer_check = ctx.get("answer_check")
        answer_eval = ctx.get("answer_evaluation")
        answer_error_notice = str(ctx.get("answer_error_notice", "") or "").strip()
        policy_format_error = str(ctx.get("policy_format_error", "") or "").strip()

        if policy_format_error and not (isinstance(answer_eval, dict) and answer_eval.get("has_answer")):
            return (
                "Policy format violation: non-final turn must include exactly one valid "
                "<tool_call>{\"name\":\"...\",\"arguments\":{...}}</tool_call>. "
                "Retry with one concrete CRUD action only."
            )

        # Hard guard for malformed final-answer format to ensure immediate correction signal.
        if isinstance(answer_eval, dict) and answer_eval.get("has_answer") and not answer_eval.get("format_ok", False):
            return (
                "VERDICT: INCORRECT\n"
                "REASON: Final answer format is invalid.\n"
                "FEEDBACK: Output exactly <answer>\\boxed{final_answer}</answer> with no extra text."
            )

        text_blocks = [
            f"Question: {self.instruction}",
        ]
        if self.target_canvas:
            text_blocks.append(
                "Target canvas metadata: " + json.dumps(self.target_canvas, ensure_ascii=False)
            )
        if isinstance(answer_eval, dict) and answer_eval.get("has_answer"):
            text_blocks.append(
                "Answer evaluation turn: return exactly VERDICT/REASON/FEEDBACK and judge semantic equivalence."
            )
        if answer_error_notice:
            text_blocks.append(answer_error_notice)
        if isinstance(answer_check, dict):
            text_blocks.append(
                "Gold answers: " + json.dumps(answer_check.get("gold_answers", []), ensure_ascii=False)
            )
            text_blocks.append(
                "Model boxed answer: " + str(answer_check.get("model_answer_boxed", ""))
            )
            text_blocks.append(
                "Answer format ok: " + str(bool(answer_check.get("answer_format_ok", False)))
            )

        image_urls = []
        target_ref = self.target_image_url or target_image_path
        if target_ref:
            image_urls.append(target_ref)
        rendered_ref = rendered_image_url or rendered_image_path
        if rendered_ref:
            image_urls.append(rendered_ref)

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
        raw_output = self._generate_once(messages)
        if isinstance(answer_eval, dict) and answer_eval.get("has_answer"):
            return self._sanitize_answer_feedback(raw_output)
        return self._sanitize_non_answer_feedback(raw_output)

    def get_total_cost(self) -> float:
        return self.total_cost

    def get_last_usage(self) -> Dict[str, int]:
        return dict(self.last_usage)


def load_user(
    strategy: str,
    model: str,
    provider: str,
    api_base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    max_tokens: Optional[int] = None,
) -> BaseUserSimulation:
    strategy = strategy.lower()
    if strategy == "human":
        return HumanUserSimulation()
    if strategy == "llm":
        return LLMUserSimulation(
            model=model,
            provider=provider,
            api_base_url=api_base_url,
            api_key=api_key,
            max_tokens=max_tokens,
        )
    if strategy == "scripted":
        return ScriptedUserSimulation()
    raise ValueError(f"Unknown user strategy: {strategy}")
