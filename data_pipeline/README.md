# Data Pipeline (Canvas Multi-turn SFT)

Standalone project under `data_pipeline/`, no runtime import dependency on `Canvas-CoT/`.

## Folder Structure

```text
data_pipeline/
  blueprint_pipeline.py          # stage-1 entrypoint
  rollout_pipeline.py            # stage-2 entrypoint
  examples_seed.jsonl
  model_backends/
    base.py
    api_backend.py
    local_backend.py
  blueprints/
    prompts.py
    schema.py
    execution.py
    io_utils.py
    generator.py
    review.py
    cli.py
  rollouts/
    prompts.py
    parser.py
    model_client.py
    checker.py
    collector.py
    cli.py
  canvas_env/
    environment.py
    tool_registry.py
    tools/
      base.py
      insert_element.py
      modify_element.py
      replace_element.py
      remove_element.py
      clear.py
    blackboard.py
```

## Stage-1: Blueprint Generation

Pipeline:

1. Model generation (API or local model)
2. Schema validation
3. APIGen-style execution validation (optional_gt_action_plan dry-run in Canvas env)
4. Retry on invalid JSON/schema/execution
5. Optional committee review
6. Save accepted/rejected JSONL

Image-base policy (enforced):
- Stage-1 canonicalizes blueprint canvas to image-based init svg.
- `init_svg` is always normalized to `main_svg` + `base_image` (input image background).
- All success targets should be overlay elements edited on top of `base_image`.

### Input JSONL (seed)

Each line should be JSON, minimal recommended fields:

```json
{"task_id":"t001","question":"...","image_path":"...","init_svg":"<svg id='main_svg'>...</svg>","answer":"<answer>\\boxed{...}</answer>"}
```

`image_path` is required for image-based canvas. Relative paths are resolved against the seed jsonl directory.

### Output JSONL (accepted)

```json
{
  "task_id":"t001",
  "blueprint": { "...": "..." },
  "answer":"<answer>\\boxed{...}</answer>",
  "reference_answer":"...",
  "seed_meta":{"source_index":0},
  "review":{"enabled":false},
  "stage1_checks":{"exec_check":{"enabled":true,"valid":true}}
}
```

### Run Stage-1 (API)

```bash
python /m2/slz/lrt/data_pipeline/blueprint_pipeline.py \
  --input-jsonl /m2/slz/lrt/data_pipeline/examples_seed.jsonl \
  --output-jsonl /m2/slz/lrt/data_pipeline/blueprints.jsonl \
  --rejected-jsonl /m2/slz/lrt/data_pipeline/blueprints_rejected.jsonl \
  --backend api \
  --model gpt-4o \
  --base-url "$BASE_URL" \
  --api-key "$OPENAI_API_KEY" \
  --max-retries 2 \
  --enable-exec-check \
  --require-gt-action-plan \
  --max-plan-steps 12
```

Index slicing options (left-closed, right-open) are supported in stage-1:

```bash
--start 100 --end 200
```

### Run Stage-1 (Local)

```bash
python /m2/slz/lrt/data_pipeline/blueprint_pipeline.py \
  --input-jsonl /m2/slz/lrt/data_pipeline/examples_seed.jsonl \
  --output-jsonl /m2/slz/lrt/data_pipeline/blueprints_local.jsonl \
  --backend local \
  --model /path/to/your-local-model \
  --local-device auto \
  --local-max-new-tokens 1024
```

## Stage-2: Rollout Collection

Use stage-1 blueprints to collect multi-turn trajectories with `<think>` + `<tool_call>`, execute Canvas CRUD tools, and write training-ready conversations.
Critic-in-the-loop is always enabled (`policy -> tool/env -> critic feedback -> policy`).

Image-base policy (enforced):
- Stage-2 runtime normalizes/repairs `init_svg` before rollout.
- Canvas always starts from input image as `base_image` inside `main_svg`.
- CRUD operations are expected to edit overlay elements on top of the base image.

Supported tools in runtime:

- `insert_element`
- `modify_element`
- `replace_element`
- `remove_element`
- `clear`

### Input JSONL

Either of:

1. stage-1 accepted row: `{"task_id":"...","blueprint":{...}}`
2. direct blueprint row: `{...blueprint fields...}`

### Output JSONL (accepted)

Each line includes:

- `task_id`
- `success` / `status` / `finish_reason`
- `blueprint`
- `turns` (assistant output, parsed tool calls, tool execution logs)
- `sft_messages` (multi-turn training conversation)
- `final_canvas_state`
- `evaluation` (rule checks from `success_check`)
- `critic` (enabled/call count/reward summary/raw critic outputs)

Answer correctness is a hard requirement in stage-2:
- stage-2 first reads `answer` (required format: `<answer>\\boxed{...}</answer>`), then falls back to `reference_answer`.
- predicted final answer must appear in `<answer>...</answer>` and contain `\\boxed{...}` (this structure is hard-required).
- extra text outside `<answer>` is tolerated; correctness check only focuses on boxed payload.
- checker first does strict normalized exact match, then (default on) uses LLM semantic equivalence judge for boxed content.
- equivalent forms like `1/2` vs `0.5` can pass via the LLM judge.
- `evaluation.answer_check.pass` must be `true` for `evaluation.pass=true`.
- If both `answer` and `reference_answer` are missing, `evaluation.pass` will be `false`.

Stage-2 hard filter is always enabled:
- passed rollouts are written to `output-jsonl`;
- failed rollouts are written to `rejected-jsonl` (or auto file `<output>_filtered_out.jsonl` if not provided).
- rollout must contain meaningful interaction:
  - at least one tool call,
  - at least one successful tool execution,
  - at least one critic call.

### Run Stage-2 (API)

```bash
python /m2/slz/lrt/data_pipeline/rollout_pipeline.py \
  --input-jsonl /m2/slz/lrt/data_pipeline/blueprints.jsonl \
  --output-jsonl /m2/slz/lrt/data_pipeline/rollouts.jsonl \
  --rejected-jsonl /m2/slz/lrt/data_pipeline/rollouts_rejected.jsonl \
  --backend api \
  --model gpt-4o \
  --base-url "$BASE_URL" \
  --api-key "$OPENAI_API_KEY" \
  --render-dir /m2/slz/lrt/data_pipeline/renders \
  --max-tools-per-turn 8 \
  --force-final-retries 2 \
  --enable-answer-judge \
  --answer-judge-max-tokens 256
```

### Critic Config

Critic is always on. By default it uses the same backend/model as policy.
You can still set critic backend/model independently:

```bash
--critic-backend local \
--critic-model /path/to/critic-model \
--critic-local-device auto
```

### Run Stage-2 (Local)

```bash
python /m2/slz/lrt/data_pipeline/rollout_pipeline.py \
  --input-jsonl /m2/slz/lrt/data_pipeline/blueprints_local.jsonl \
  --output-jsonl /m2/slz/lrt/data_pipeline/rollouts_local.jsonl \
  --backend local \
  --model /path/to/your-local-model \
  --local-device auto \
  --local-max-new-tokens 1024
```

### Notes

- Stage-2 requires `bs4` + `lxml` + `playwright` to initialize Canvas runtime.
- Stage-1 schema currently requires `image_path` (single image string).
- Stage-2 collector also accepts optional `image_paths` list if you manually extend blueprint rows.
- Critic args omitted => critic defaults to policy backend/model/api settings.
- Tool calls always produce rendered observations (default render dir: `/tmp/canvas_rollout_renders`).
- Stage-2 now follows Canvas-style feedback:
  - successful tool call -> append rendered notebook image as next user observation;
  - failed tool call -> append `<tool_response>...</tool_response>` text;
  - critic is triggered only when this round has successful rendered tool output;
  - critic feedback -> append `<tool_response><image>This is the state of notebook. Critical Check: ...</tool_response>`.
- Assistant history for next-turn runtime drops `<think>` (Canvas-style), while `sft_messages` still keep raw assistant output for training data.
