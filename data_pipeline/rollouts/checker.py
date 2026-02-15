from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional

ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", flags=re.IGNORECASE | re.DOTALL)


def _id_exists(html: str, element_id: str) -> bool:
    pattern = re.compile(rf"id\s*=\s*['\"]{re.escape(element_id)}['\"]")
    return pattern.search(html) is not None


def _attr_equals(html: str, element_id: str, attr: str, value: str) -> bool:
    tag_pattern = re.compile(rf"<[^>]*id\s*=\s*['\"]{re.escape(element_id)}['\"][^>]*>")
    m = tag_pattern.search(html)
    if not m:
        return False
    tag = m.group(0)
    attr_pattern = re.compile(rf"{re.escape(attr)}\s*=\s*['\"]{re.escape(value)}['\"]")
    return attr_pattern.search(tag) is not None


def _extract_boxed_contents(text: str) -> List[str]:
    out: List[str] = []
    if not text:
        return out

    marker = "\\boxed{"
    idx = 0
    while True:
        start = text.find(marker, idx)
        if start < 0:
            break
        i = start + len(marker)
        depth = 1
        while i < len(text) and depth > 0:
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            i += 1
        if depth == 0:
            out.append(text[start + len(marker) : i - 1].strip())
            idx = i
        else:
            break
    return out


def _extract_answer_bodies(text: str) -> List[str]:
    out: List[str] = []
    if not text:
        return out
    for m in ANSWER_PATTERN.finditer(text):
        out.append(m.group(1).strip())
    return out


def _normalize_answer(text: str) -> str:
    s = (text or "").strip()
    if len(s) >= 2 and s[0] == "$" and s[-1] == "$":
        s = s[1:-1].strip()
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("（", "(").replace("）", ")")
    s = s.replace("−", "-")
    s = re.sub(r"\s+", "", s).lower()
    return s


def _split_reference_candidates(reference_answer: str) -> List[str]:
    ref = (reference_answer or "").strip()
    if not ref:
        return []
    parts = re.split(r"\s*(?:\bor\b|或|；|;|\|\|)\s*", ref, flags=re.IGNORECASE)
    candidates = [x.strip() for x in parts if x.strip()]
    return candidates if candidates else [ref]


def _extract_reference_candidates(reference_answer: str) -> List[str]:
    ref = (reference_answer or "").strip()
    if not ref:
        return []

    m = re.search(r"<answer>(.*?)</answer>", ref, flags=re.IGNORECASE | re.DOTALL)
    if m:
        body = m.group(1).strip()
    else:
        body = ref

    if "\\boxed{" in body:
        boxed = _extract_boxed_contents(body)
        if boxed:
            return boxed

    return _split_reference_candidates(body)


def _evaluate_answer_correct(
    reference_answer: Optional[str],
    assistant_outputs: List[str],
    question: str = "",
    answer_judge: Optional[Callable[[str, List[str], str], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    ref = (reference_answer or "").strip()
    if not ref:
        return {
            "required": True,
            "pass": False,
            "reason": "missing_answer_reference",
            "reference_answer": "",
            "reference_candidates": [],
            "predicted_boxed": "",
            "predicted_boxed_normalized": "",
            "has_boxed": any("\\boxed{" in x for x in assistant_outputs),
            "has_answer_tag": any(bool(_extract_answer_bodies(x)) for x in assistant_outputs),
            "has_boxed_in_answer": any(
                bool(_extract_boxed_contents(body))
                for x in assistant_outputs
                for body in _extract_answer_bodies(x)
            ),
            "judge_result": {},
        }

    answer_bodies: List[str] = []
    boxed_candidates: List[str] = []
    for output in assistant_outputs:
        bodies = _extract_answer_bodies(output)
        answer_bodies.extend(bodies)
        for body in bodies:
            boxed_candidates.extend(_extract_boxed_contents(body))

    has_answer_tag = bool(answer_bodies)
    has_boxed_in_answer = bool(boxed_candidates)
    predicted = boxed_candidates[-1].strip() if boxed_candidates else ""
    pred_norm = _normalize_answer(predicted)
    ref_candidates = _extract_reference_candidates(ref)
    ref_norm_candidates = [_normalize_answer(x) for x in ref_candidates]
    is_pass = pred_norm in ref_norm_candidates and pred_norm != ""
    judge_result: Dict[str, Any] = {}
    llm_equivalent = False
    if (not is_pass) and pred_norm and ref_candidates and answer_judge is not None:
        try:
            judge_result = answer_judge(question, ref_candidates, predicted)
            llm_equivalent = bool(judge_result.get("equivalent", False))
        except Exception as exc:
            judge_result = {"equivalent": False, "reason": f"answer_judge_error:{type(exc).__name__}:{exc}"}
        if llm_equivalent:
            is_pass = True
    reason = ""
    if not has_answer_tag:
        reason = "answer_tag_missing"
    elif not has_boxed_in_answer:
        reason = "boxed_missing_in_answer"
    elif llm_equivalent:
        reason = "llm_equivalent"
    elif not is_pass:
        reason = "answer_incorrect_or_missing"

    return {
        "required": True,
        "pass": is_pass,
        "reason": reason,
        "reference_answer": ref,
        "reference_candidates": ref_candidates,
        "reference_candidates_normalized": ref_norm_candidates,
        "predicted_boxed": predicted,
        "predicted_boxed_normalized": pred_norm,
        "has_boxed": any("\\boxed{" in x for x in assistant_outputs),
        "has_answer_tag": has_answer_tag,
        "has_boxed_in_answer": has_boxed_in_answer,
        "judge_result": judge_result,
    }


def evaluate_success(
    blueprint: Dict[str, Any],
    final_state_html: str,
    assistant_outputs: List[str],
    reference_answer: Optional[str] = None,
    question: str = "",
    answer_judge: Optional[Callable[[str, List[str], str], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    sc = blueprint.get("success_check", {})

    id_results = []
    for element_id in sc.get("must_have_ids", []):
        ok = _id_exists(final_state_html, element_id)
        id_results.append({"id": element_id, "pass": ok})

    attr_results = []
    for item in sc.get("must_have_attrs", []):
        if not isinstance(item, dict):
            attr_results.append({"item": item, "pass": False, "error": "invalid item"})
            continue
        ok = _attr_equals(final_state_html, item.get("id", ""), item.get("attr", ""), item.get("value", ""))
        attr_results.append({"item": item, "pass": ok})

    need_boxed = bool(sc.get("need_boxed_answer", False))
    boxed_pass = True
    answer_check = _evaluate_answer_correct(
        reference_answer=reference_answer,
        assistant_outputs=assistant_outputs,
        question=question,
        answer_judge=answer_judge,
    )
    answer_tag_pass = bool(answer_check.get("has_answer_tag", False))

    if need_boxed:
        boxed_pass = bool(answer_check.get("has_boxed_in_answer", False))

    passed = (
        all(x["pass"] for x in id_results)
        and all(x["pass"] for x in attr_results)
        and boxed_pass
        and bool(answer_check.get("pass", False))
    )

    return {
        "pass": passed,
        "must_have_ids": id_results,
        "must_have_attrs": attr_results,
        "need_boxed_answer": need_boxed,
        "boxed_pass": boxed_pass,
        "answer_tag_pass": answer_tag_pass,
        "answer_check": answer_check,
    }
