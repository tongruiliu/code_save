# canvas_tau_bench

A standalone minimal implementation independent of `tau-bench`:
- Keeps a **multi-turn policy-vs-critic loop** for distillation data generation.
- Uses **Canvas CRUD** operations (`insert/modify/replace/remove/clear/finish`) as the action space.
- Uses tau-bench-style reward computation (state-hash comparison against replayed target actions).
- Exports assistant-turn SFT records in JSONL format.

## Structure

- `run.py`: Entry script, benchmark execution, and SFT JSONL export.
- `agent.py`: Policy agent that parses `<think>/<tool>/<answer>` from assistant outputs.
- `env.py`: Canvas CRUD environment, critic feedback loop, and reward logic.
- `tools.py`: Canvas CRUD tool definitions and execution.
- `user.py`: Critic simulator with `scripted / human / llm` strategies.
- `types.py`: Data structure definitions.

## Role Protocol

- `assistant`: The policy model being distilled.
- `user`: The critic providing turn-level feedback.
- The assistant is expected to output:
  - Non-final turns should include `<think>...</think>` and advance only one step
  - `<tool>{"name":"...","args":{...}}</tool>` when executing one CRUD action for that step
  - Final answer only as `<answer>\boxed{final_answer}</answer>` (also accepts `/boxed{...}` for compatibility)
  - Do not include extra text inside `<answer>` beyond the boxed payload
  - Do not include extra text outside `<answer>` in the final turn
- The environment executes the tool, compares rendered-vs-target state, and sends critic feedback back as `user` content.
- Episode termination is answer-gated:
  - Critic decides correctness (not hard-coded output matching).
  - Stop only when `<answer>` appears and critic returns `VERDICT: CORRECT`.
  - If `<answer>` appears but is wrong, critic receives target render, rendered result, gold answers, model answer, and hallucination-check notice.
  - For SFT trajectory export, terminal samples are stored with assistant as the last speaker; critic verdict remains in turn metadata.

Example assistant turn:

```text
<think>I should insert the first node under root.</think>
<tool>{"name":"insert_element","args":{"fragment":"<div id='plan'>draft</div>","rootId":"root"}}</tool>
```

Example final answer:

```text
<answer>\boxed{done}</answer>
```

Critic decision format for answer turns:

```text
VERDICT: CORRECT or VERDICT: INCORRECT
REASON: ...
FEEDBACK: ...
```

## Install

```bash
cd /m2/slz/lrt/canvas_tau_bench
pip install -r requirements.txt
```

## Run

Recommended (module mode):

```bash
cd /m2/slz/lrt
python -m canvas_tau_bench.run \
  --model gpt-4o-mini \
  --model-provider openai \
  --user-strategy scripted \
  --num-trials 1
```

Direct script execution is also supported:

```bash
python /m2/slz/lrt/canvas_tau_bench/run.py \
  --model gpt-4o-mini \
  --model-provider openai \
  --user-strategy scripted
```

## Key Arguments

- `--model`, `--model-provider`: Main agent model configuration.
- `--user-strategy`: Critic strategy: `scripted` (default), `human`, or `llm`.
- `--user-model`, `--user-model-provider`: Configuration for the `llm` critic simulator.
- `--task-ids`: Explicit list of task IDs to run.
- `--num-trials`: Number of repeated trials.
- `--max-steps`: Maximum dialogue steps per task.
- `--log-dir`: Output directory for result files.
- `--sft-jsonl`: Optional explicit output path for SFT records.

## Notes

- This is a standalone implementation and does not modify original `tau-bench` code.
- Three demo tasks are included by default. You can replace them in `build_demo_tasks()` inside `run.py`.
- `finish_canvas` is treated as a normal tool action; only critic-accepted final answer ends the episode.
- By default, results are saved as:
  - `*.json`: full trajectories and reward metadata
  - `*.sft.jsonl`: assistant-turn SFT records (`messages` + `assistant_target` + `turn_meta`)
- Current renderer payload is JSON-based (`target_canvas` vs `rendered_canvas`). You can replace it with real image rendering while keeping the same critic loop API.
- Critic context also exposes `target_image` and `rendered_image` fields (currently JSON payload aliases) so you can swap in real image paths/bytes later.
- For answer turns, critic context includes `answer_check` with `gold_answers` and boxed model answer, enabling semantic-equivalence judgement (e.g., `0.5 == 1/2`).
