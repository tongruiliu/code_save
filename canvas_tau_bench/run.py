from __future__ import annotations

import argparse
import json
import os
import random
import traceback
from datetime import datetime
from math import comb
from typing import Any, Dict, List, Optional

from litellm import provider_list

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from canvas_tau_bench.agent import ToolCallingAgent
    from canvas_tau_bench.env import CanvasCRUDEnv
    from canvas_tau_bench.types import Action, EnvRunResult, Task
else:
    from .agent import ToolCallingAgent
    from .env import CanvasCRUDEnv
    from .types import Action, EnvRunResult, Task


def build_demo_tasks() -> List[Task]:
    return [
        Task(
            user_id="canvas_user_001",
            instruction=(
                "Complete the following canvas operations in order: "
                "1) Insert <div id='plan'>draft</div> into root; "
                "2) Modify plan text to 'step-1 complete'; "
                "3) Replace plan with <div id='final_card'>done</div>; "
                "4) Call finish_canvas after completion; "
                "5) Reply with done."
            ),
            actions=[
                Action(name="insert_element", kwargs={"fragment": "<div id='plan'>draft</div>", "rootId": "root"}),
                Action(name="modify_element", kwargs={"targetId": "plan", "attrs": {"text": "step-1 complete"}}),
                Action(name="replace_element", kwargs={"targetId": "plan", "fragment": "<div id='final_card'>done</div>"}),
                Action(name="finish_canvas", kwargs={"summary": "task completed"}),
            ],
            outputs=["done"],
        ),
        Task(
            user_id="canvas_user_002",
            instruction=(
                "Create two nodes and validate ordering: "
                "first insert <div id='a'>A</div>, then insert <div id='b'>B</div>; "
                "next insert <div id='c'>C</div> with beforeId='b'; "
                "then remove a; finally call finish_canvas and reply with done."
            ),
            actions=[
                Action(name="insert_element", kwargs={"fragment": "<div id='a'>A</div>", "rootId": "root"}),
                Action(name="insert_element", kwargs={"fragment": "<div id='b'>B</div>", "rootId": "root"}),
                Action(name="insert_element", kwargs={"fragment": "<div id='c'>C</div>", "rootId": "root", "beforeId": "b"}),
                Action(name="remove_element", kwargs={"targetId": "a"}),
                Action(name="finish_canvas", kwargs={"summary": "ordered insert and remove complete"}),
            ],
            outputs=["done"],
        ),
        Task(
            user_id="canvas_user_003",
            instruction=(
                "First insert <div id='tmp'>temp</div>, "
                "then call clear, "
                "then insert <div id='result'>ok</div>; "
                "finally call finish_canvas and reply with done."
            ),
            actions=[
                Action(name="insert_element", kwargs={"fragment": "<div id='tmp'>temp</div>", "rootId": "root"}),
                Action(name="clear", kwargs={}),
                Action(name="insert_element", kwargs={"fragment": "<div id='result'>ok</div>", "rootId": "root"}),
                Action(name="finish_canvas", kwargs={"summary": "clear flow complete"}),
            ],
            outputs=["done"],
        ),
    ]


def _coerce_base_items(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        if isinstance(payload.get("data"), list):
            return [x for x in payload["data"] if isinstance(x, dict)]
        # MathVista-style: {"1": {...}, "2": {...}, ...}
        kv: List[tuple[str, Any]] = []
        for k, v in payload.items():
            if isinstance(v, dict):
                kv.append((str(k), v))

        def key_fn(x: tuple[str, Any]) -> tuple[int, str]:
            k = x[0]
            if k.isdigit():
                return (0, f"{int(k):012d}")
            return (1, k)

        kv.sort(key=key_fn)
        return [v for _, v in kv]
    return []


def _format_choices(choices: Any) -> str:
    if not isinstance(choices, list) or len(choices) == 0:
        return ""
    lines: List[str] = []
    for i, c in enumerate(choices):
        label = chr(ord("A") + i) if i < 26 else f"C{i + 1}"
        lines.append(f"{label}. {c}")
    return "Choices:\n" + "\n".join(lines)


def build_tasks_from_base_data(
    json_path: str,
    data_root: Optional[str] = None,
    max_samples: Optional[int] = None,
    offset: int = 0,
    skip_answer_type_list: bool = True,
) -> List[Task]:
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    items = _coerce_base_items(payload)

    if offset > 0:
        items = items[offset:]

    root = data_root or os.path.dirname(json_path)
    tasks: List[Task] = []

    for i, item in enumerate(items):
        question = str(item.get("question", "")).strip()
        if not question:
            continue

        answer_type = str(item.get("answer_type", "")).strip().lower()
        if skip_answer_type_list and answer_type == "list":
            continue

        image_rel = str(item.get("image", "")).strip()
        target_image_path = ""
        if image_rel:
            if os.path.isabs(image_rel):
                target_image_path = image_rel
            else:
                target_image_path = os.path.join(root, image_rel)

        choices_text = _format_choices(item.get("choices"))
        instruction = question if not choices_text else f"{question}\n{choices_text}"

        answer = item.get("answer")
        outputs: List[str] = []
        if isinstance(answer, list):
            outputs.append(json.dumps(answer, ensure_ascii=False))
            outputs.extend(str(x) for x in answer)
        elif answer is not None:
            ans = str(answer).strip()
            if ans:
                outputs.append(ans)

        pid = str(item.get("pid") or item.get("id") or f"sample_{offset + i}")
        tasks.append(
            Task(
                user_id=f"mathvista_{pid}",
                instruction=instruction,
                actions=[],
                outputs=outputs,
                target_image_path=target_image_path or None,
            )
        )

        if max_samples is not None and len(tasks) >= max_samples:
            break

    return tasks


def display_metrics(results: List[EnvRunResult]) -> None:
    if not results:
        print("No results.")
        return

    def is_successful(reward: float) -> bool:
        return abs(reward - 1.0) <= 1e-6

    num_trials = len(set(r.trial for r in results))
    avg_reward = sum(r.reward for r in results) / len(results)

    c_per_task_id = {}
    for r in results:
        c_per_task_id.setdefault(r.task_id, 0)
        c_per_task_id[r.task_id] += 1 if is_successful(r.reward) else 0

    print(f"Average reward: {avg_reward:.4f}")
    print("Pass^k")
    for k in range(1, num_trials + 1):
        total = 0.0
        for c in c_per_task_id.values():
            total += comb(c, k) / comb(num_trials, k)
        print(f"  k={k}: {total / len(c_per_task_id):.4f}")


def build_sft_records(result: EnvRunResult) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    turns = result.info.get("turns", []) if isinstance(result.info, dict) else []
    assistant_turn_idx = 0

    for i, msg in enumerate(result.traj):
        if msg.get("role") != "assistant":
            continue
        record: Dict[str, Any] = {
            "task_id": result.task_id,
            "trial": result.trial,
            "reward": result.reward,
            "messages": result.traj[:i],
            "assistant_target": msg.get("content", ""),
        }
        if assistant_turn_idx < len(turns):
            record["turn_meta"] = turns[assistant_turn_idx]
        records.append(record)
        assistant_turn_idx += 1

    return records


def write_sft_jsonl(results: List[EnvRunResult], path: str) -> int:
    num_records = 0
    with open(path, "w", encoding="utf-8") as f:
        for result in results:
            for record in build_sft_records(result):
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                num_records += 1
    return num_records


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Standalone canvas-CRUD tau-style benchmark")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--model-provider", type=str, required=True, choices=provider_list)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-steps", type=int, default=30)

    p.add_argument("--user-strategy", type=str, default="scripted", choices=["scripted", "human", "llm"])
    p.add_argument("--user-model", type=str, default="gpt-4o-mini")
    p.add_argument("--user-model-provider", type=str, default="openai")

    p.add_argument("--num-trials", type=int, default=1)
    p.add_argument("--task-ids", type=int, nargs="+", default=None)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--log-dir", type=str, default="results")
    p.add_argument("--sft-jsonl", type=str, default=None)
    p.add_argument("--base-data-json", type=str, default=None)
    p.add_argument("--base-data-root", type=str, default=None)
    p.add_argument("--base-max-samples", type=int, default=None)
    p.add_argument("--base-offset", type=int, default=0)
    p.add_argument("--skip-answer-type-list", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    if args.base_data_json:
        tasks = build_tasks_from_base_data(
            json_path=args.base_data_json,
            data_root=args.base_data_root,
            max_samples=args.base_max_samples,
            offset=args.base_offset,
            skip_answer_type_list=args.skip_answer_type_list,
        )
        if len(tasks) == 0:
            raise ValueError(
                f"No tasks loaded from base data: {args.base_data_json}. "
                "Check path/format or relax filters."
            )
        print(f"Loaded {len(tasks)} tasks from base data: {args.base_data_json}")
    else:
        tasks = build_demo_tasks()

    if args.task_ids:
        for idx in args.task_ids:
            if idx < 0 or idx >= len(tasks):
                raise ValueError(f"Invalid task id: {idx}")
        task_ids = list(args.task_ids)
    else:
        task_ids = list(range(len(tasks)))

    os.makedirs(args.log_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = os.path.join(args.log_dir, f"canvas-crud-{args.model.split('/')[-1]}-{ts}.json")

    template_env = CanvasCRUDEnv(
        tasks=tasks,
        user_strategy=args.user_strategy,
        user_model=args.user_model,
        user_provider=args.user_model_provider,
    )
    agent = ToolCallingAgent(
        tools_info=template_env.tools_info,
        wiki=template_env.wiki,
        model=args.model,
        provider=args.model_provider,
        temperature=args.temperature,
    )

    results: List[EnvRunResult] = []

    for trial in range(args.num_trials):
        current_ids = list(task_ids)
        if args.shuffle:
            random.shuffle(current_ids)

        for idx in current_ids:
            env = CanvasCRUDEnv(
                tasks=tasks,
                user_strategy=args.user_strategy,
                user_model=args.user_model,
                user_provider=args.user_model_provider,
            )
            try:
                solve_res = agent.solve(env=env, task_index=idx, max_num_steps=args.max_steps)
                result = EnvRunResult(
                    task_id=idx,
                    reward=solve_res.reward,
                    info=solve_res.info,
                    traj=solve_res.messages,
                    trial=trial,
                )
            except Exception as exc:
                result = EnvRunResult(
                    task_id=idx,
                    reward=0.0,
                    info={"error": str(exc), "traceback": traceback.format_exc()},
                    traj=[],
                    trial=trial,
                )

            print(("PASS" if result.reward == 1 else "FAIL"), f"task_id={idx}", f"reward={result.reward}")
            results.append(result)

    display_metrics(results)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in results], f, ensure_ascii=False, indent=2)

    sft_path = args.sft_jsonl
    if sft_path is None:
        sft_path = out_path.replace(".json", ".sft.jsonl")
    num_records = write_sft_jsonl(results, sft_path)

    print(f"\nSaved results to {out_path}")
    print(f"Saved {num_records} SFT records to {sft_path}\n")


if __name__ == "__main__":
    main()
