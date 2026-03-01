from __future__ import annotations

import json
from typing import Any, Dict, List


SYSTEM_PROMPT_TEMPLATE = """
# Objective
You are a Visual-Reasoning Agent. Solve the question by synchronizing reasoning with a notebook.

# Output Protocol
1) Think one step:
<think>...</think>

2) If notebook update is needed, emit one or more tool calls:
<tool_call>{"name":"tool_name","arguments":{...}}</tool_call>

3) Once confident, provide final answer:
<answer>\\boxed{{final_answer}}</answer>

Rules:
- Use incremental notebook updates.
- Use valid JSON for each <tool_call> block.
- Allowed tools: insert_element, modify_element, remove_element, replace_element, clear.
- Tool schema:
<tools>
{tools_json}
</tools>
""".strip()


CRITIQUE_SYSTEM = """
You are a strict visual reasoning critic.
Given Question + Original Image + Notebook State image:
- Identify visual/reasoning errors.
- Return concise feedback for next step.
- Do not solve the full problem.
""".strip()


CRITIQUE_PROMPT = "<tool_response><image>This is the notebook state. Critical Check: {critical_check}</tool_response>"


FINAL_ANSWER_RETRY_PROMPT = (
    "Please provide the final answer directly. Output only: "
    "<answer>\\boxed{...}</answer>"
)


def build_system_prompt(tools: List[Dict[str, Any]]) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(tools_json=json.dumps(tools, ensure_ascii=False, indent=2))
