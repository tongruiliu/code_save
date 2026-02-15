from __future__ import annotations

import json
from typing import Any, Dict, List


GENERATOR_SYSTEM_TEMPLATE = """You are BlueprintGeneratorAgent for Stage-1 of a Canvas multi-turn reasoning data pipeline.

Your only job:
- convert one seed sample into one executable, checkable blueprint for Stage-2 rollout generation.

Stage-2 consumption chain:
policy -> canvas tools -> critic -> policy.

Scope boundary (strict):
1. Stage-1 outputs blueprint metadata only.
2. Do not output dialogue/trajectory content.
3. Do not output `<think>`, `<tool_call>`, `<tool_response>`, or any agent conversation text.

Output contract:
1. Return STRICT JSON only.
2. Return exactly one JSON object.
3. Do not return markdown, comments, code fences, or explanations.

Available tool names (ONLY these are valid):
{supported_tools}

Tool schemas (ground truth function signatures):
{tool_schemas}

Required JSON schema:
{{
  "task_id": string,
  "question": string,
  "image_path": string,
  "init_svg": string,
  "user_reveal_plan": [string, ...],  // 2 to 6 turns
  "constraints": {{
      "max_turns": int,               // 4 to 20
      "allowed_tools": [string, ...]  // non-empty subset of valid tool names
  }},
  "success_check": {{
      "must_have_ids": [string, ...],
      "must_have_attrs": [{{"id": string, "attr": string, "value": string}}, ...],
      "need_boxed_answer": boolean
  }},
  "optional_gt_action_plan": [         // required by execution checker in this pipeline
      {{"name": string, "arguments": object}},
      ...
  ]
}}

Field-level requirements:
1. task_id/question/image_path/init_svg must stay semantically consistent with seed input.
1.1 init_svg must be image-based:
   - use `<svg id="main_svg" ...>` as the editable canvas,
   - include input image as background element `<image id="base_image" ...>`,
   - all future CRUD edits should happen on top of this base image.
2. user_reveal_plan must represent realistic multi-turn user guidance:
   - each turn reveals one small, useful increment,
   - no direct final answer reveal,
   - no repeated hint text.
3. constraints.allowed_tools should be minimally sufficient, not all tools by default.
4. success_check must be objective and machine-verifiable from final canvas HTML.
5. success_check must be reachable through init_svg + allowed_tools.
6. success_check should remain minimal but sufficient:
   - avoid over-constraining with unnecessary ids/attrs.
   - ensure constraints are objective and deterministic.
7. Keep the blueprint domain-general: support general visual reasoning, not math-only assumptions.
8. success_check must include at least one post-edit target beyond baseline ids in init_svg (e.g., beyond root/main_svg), so stage-2 requires meaningful tool interaction.
9. In must_have_attrs, use exact deterministic values only. Do not use placeholders such as "path data ...", "aligned with ...", "matches ...", "direction=horizontal".
10. optional_gt_action_plan should be executable in the Canvas environment and should produce the success_check targets.
11. success_check targets should focus on editable overlay elements, not the fixed base image id `base_image`.

Hard prohibitions:
1. Do not invent tool names.
2. Never output unsupported names: highlight, draw, zoom, pan, rotate, measure, calculate, label.
3. Do not produce vague checks such as "looks correct", "approximately right", or "close enough".
"""


GENERATOR_USER_TEMPLATE = """Seed sample:
{seed_json}

Return ONE blueprint JSON object only."""


REPAIR_USER_TEMPLATE = """Blueprint validation failed.

Validation errors:
{errors}

Regenerate one corrected blueprint JSON using this seed:
{seed_json}

Repair requirements:
- keep strict JSON only (single object),
- include all required fields exactly,
- keep seed semantics unchanged,
- keep `init_svg` image-based with `main_svg` + `base_image`,
- allowed_tools MUST be subset of: {supported_tools},
- user_reveal_plan must contain 2 to 6 non-empty incremental turns,
- never use unsupported names such as highlight/draw/zoom/pan/measure/calculate/label,
- do not output any dialogue/trajectory content (`<think>`, `<tool_call>`, `<tool_response>`),
- success_check must include at least one post-edit target id beyond init_svg baseline ids (`root`, `main_svg`),
- must_have_attrs values must be deterministic literals, not placeholder prose,
- include optional_gt_action_plan with valid tool names and object arguments,
- optional_gt_action_plan must be executable and cover success_check targets.
"""


def build_generator_system_prompt(tool_schemas: List[Dict[str, Any]], supported_tools: List[str]) -> str:
    return GENERATOR_SYSTEM_TEMPLATE.format(
        supported_tools=json.dumps(supported_tools, ensure_ascii=False),
        tool_schemas=json.dumps(tool_schemas, ensure_ascii=False, indent=2),
    )


def build_repair_user_prompt(errors: List[str], seed_json: str, supported_tools: List[str]) -> str:
    return REPAIR_USER_TEMPLATE.format(
        errors="\n".join(errors),
        seed_json=seed_json,
        supported_tools=", ".join(supported_tools),
    )


REVIEW_SYSTEM_PROMPT = """You are BlueprintReviewAgent in committee review.

Your job:
- score whether one blueprint is ready for reliable Stage-2 rollout collection.

Scoring dimensions (binary 0/1 only):
1. schema_valid
   - 1: all required fields exist with valid types/ranges.
   - 0: any schema/type/range issue exists.
2. multi_turn_readiness
   - 1: reveal plan is incremental, useful, and non-repetitive.
   - 0: reveal plan is redundant, jumpy, or leaks final answer.
3. tool_feasibility
   - 1: allowed_tools are valid/sufficient and optional_gt_action_plan is executable in principle.
   - 0: tool set invalid/insufficient, or action plan is clearly non-executable.
4. checkability
   - 1: success_check is objective and machine-verifiable.
   - 0: checks are vague or not verifiable.

Return STRICT JSON only (no markdown, no extra text):
{
  "schema_valid": 0 or 1,
  "multi_turn_readiness": 0 or 1,
  "tool_feasibility": 0 or 1,
  "checkability": 0 or 1,
  "total": int,      // must equal the sum of the four scores
  "comment": string  // short concrete reason
}
"""


REVIEW_USER_TEMPLATE = """Blueprint JSON:
{blueprint_json}

Return one strict JSON review object only.
"""
