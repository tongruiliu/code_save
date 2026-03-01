# canvas_sft_pipeline

A fresh, standalone Canvas-CoT-style pipeline for generating SFT trajectories.

## Goals
- Keep Canvas reasoning flow: think -> tool call(s) -> render -> critique -> next step.
- Generate high-quality assistant-turn SFT JSONL.
- Run on MathVista slices via `[start, end)`.

## Structure
- `run.py`: CLI entrypoint.
- `pipeline.py`: core loop, tool execution, critique, rendering, SFT record export.
- `blackboard.py`: notebook rendering backend (adapted from Canvas-CoT).
- `parser.py`: robust parser for `<think>/<tool_call>/<answer>`.
- `prompts.py`: policy/critique prompts.
- `io_utils.py`: task loading and answer equivalence.
- `blackboard_tools.py`: tool schema injected into prompt.

## Install
```bash
# run from repository root (the directory containing `canvas_sft_pipeline/` and `data/`)
pip install -r canvas_sft_pipeline/requirements.txt
python -m playwright install chromium
```

## Run
```bash
# run from repository root

export POLICY_API_KEY="your_api_key"
export POLICY_API_BASE_URL="https://your-openai-compatible-endpoint/v1"
export POLICY_MODEL="gpt-5.2"
export CRITIC_MODEL="gpt-4.1-mini"
export START=0
export END=100

bash ./run_mathvista_pipeline.sh
```

Notes:
- `POLICY_API_KEY` / `POLICY_API_BASE_URL` are shared by default with critic.
- `CRITIC_MODEL` must be set explicitly and is independent from `POLICY_MODEL`.

Outputs:
- `canvas_sft_pipeline/results/canvas-sft-<tag>.json`
- `canvas_sft_pipeline/results/canvas-sft-<tag>.sft.jsonl`
- `canvas_sft_pipeline/results/renders/<session_id>/*.png`
