from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from ..model_backends import ModelBackend

from .prompts import REVIEW_SYSTEM_PROMPT, REVIEW_USER_TEMPLATE
from .schema import extract_json_block


def review_blueprint(
    backend: ModelBackend,
    blueprint: Dict[str, Any],
    committee_size: int,
    min_total_score: int,
    temperature: float,
) -> Tuple[bool, List[Dict[str, Any]]]:
    reviews: List[Dict[str, Any]] = []
    pass_votes = 0

    for _ in range(committee_size):
        user_prompt = REVIEW_USER_TEMPLATE.format(
            blueprint_json=json.dumps(blueprint, ensure_ascii=False, indent=2)
        )
        raw = backend.generate(REVIEW_SYSTEM_PROMPT, user_prompt, temperature=temperature)
        try:
            review = json.loads(extract_json_block(raw))
        except Exception:
            review = {
                "schema_valid": 0,
                "multi_turn_readiness": 0,
                "tool_feasibility": 0,
                "checkability": 0,
                "total": 0,
                "comment": "invalid review json",
            }
        reviews.append(review)
        total = review.get("total", 0)
        if isinstance(total, int) and total >= min_total_score:
            pass_votes += 1

    accepted = pass_votes >= (committee_size // 2 + 1)
    return accepted, reviews
