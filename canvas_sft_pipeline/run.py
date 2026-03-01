from __future__ import annotations

import argparse
import json
import os
import random
from datetime import datetime
from typing import Any, Dict, List

from litellm import provider_list

from .io_utils import load_mathvista_tasks
from .pipeline import ModelConfig, PipelineConfig, run_tasks


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Canvas-CoT-style SFT pipeline for MathVista")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--model-provider", type=str, required=True, choices=provider_list)
    p.add_argument("--api-key", type=str, default="")
    p.add_argument("--api-base-url", type=str, default="")
    p.add_argument("--policy-max-tokens", type=int, default=None)
    p.add_argument("--temperature", type=float, default=0.0)

    p.add_argument("--critic-model", type=str, default=None)
    p.add_argument("--critic-model-provider", type=str, default=None)
    p.add_argument("--critic-api-key", type=str, default="")
    p.add_argument("--critic-api-base-url", type=str, default="")
    p.add_argument("--critic-max-tokens", type=int, default=None)
    p.add_argument("--disable-critic", action="store_true")

    p.add_argument("--base-data-json", type=str, required=True)
    p.add_argument("--base-data-root", type=str, default=None)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int, default=3)
    p.add_argument("--skip-answer-type-list", action="store_true")

    p.add_argument("--max-rounds", type=int, default=6)
    p.add_argument("--max-final-retries", type=int, default=2)

    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--shuffle", action="store_true")

    p.add_argument("--output-dir", type=str, default="canvas_sft_pipeline/results")
    p.add_argument("--run-tag", type=str, default=None)
    return p.parse_args()


def _sum_usage(turns: List[Dict[str, Any]]) -> Dict[str, int]:
    p = 0
    c = 0
    t = 0
    for x in turns:
        u = x.get("policy_usage") or {}
        p += int(u.get("prompt_tokens", 0) or 0)
        c += int(u.get("completion_tokens", 0) or 0)
        t += int(u.get("total_tokens", 0) or 0)
    return {"prompt_tokens": p, "completion_tokens": c, "total_tokens": t}


def main() -> None:
    args = parse_args()
    if args.end <= args.start:
        raise ValueError(f"Invalid [start, end): [{args.start}, {args.end})")

    random.seed(args.seed)

    tasks = load_mathvista_tasks(
        json_path=args.base_data_json,
        data_root=args.base_data_root,
        start=args.start,
        end=args.end,
        skip_answer_type_list=args.skip_answer_type_list,
    )
    if not tasks:
        raise ValueError("No tasks loaded for given slice/filter.")

    if args.shuffle:
        random.shuffle(tasks)

    run_tag = args.run_tag or datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    render_root = os.path.join(output_dir, "renders")

    policy = ModelConfig(
        model=args.model,
        provider=args.model_provider,
        api_key=args.api_key,
        api_base_url=args.api_base_url,
        max_tokens=args.policy_max_tokens,
        temperature=args.temperature,
    )

    critic = None
    if not args.disable_critic:
        critic = ModelConfig(
            model=args.critic_model or args.model,
            provider=args.critic_model_provider or args.model_provider,
            api_key=args.critic_api_key or args.api_key,
            api_base_url=args.critic_api_base_url or args.api_base_url,
            max_tokens=args.critic_max_tokens,
            temperature=0.0,
        )

    cfg = PipelineConfig(
        policy=policy,
        critic=critic,
        max_rounds=args.max_rounds,
        max_final_retries=args.max_final_retries,
        render_root=render_root,
        output_dir=output_dir,
    )

    results = run_tasks(tasks, cfg)

    out_json = os.path.join(output_dir, f"canvas-sft-{run_tag}.json")
    out_jsonl = os.path.join(output_dir, f"canvas-sft-{run_tag}.sft.jsonl")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    total_records = 0
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r in results:
            for rec in r.get("sft_records", []):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total_records += 1

    avg_reward = sum(float(x.get("reward", 0.0) or 0.0) for x in results) / max(len(results), 1)
    pass_cnt = sum(1 for x in results if float(x.get("reward", 0.0) or 0.0) >= 1.0)

    print("\n[Summary]")
    print(f"tasks={len(results)} pass={pass_cnt} avg_reward={avg_reward:.4f}")
    print(f"results_json={out_json}")
    print(f"sft_jsonl={out_jsonl} records={total_records}")

    usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for r in results:
        u = _sum_usage(r.get("turns", []))
        usage_total["prompt_tokens"] += u["prompt_tokens"]
        usage_total["completion_tokens"] += u["completion_tokens"]
        usage_total["total_tokens"] += u["total_tokens"]
    print(f"usage_total={usage_total}")


if __name__ == "__main__":
    main()
