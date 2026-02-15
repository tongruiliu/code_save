from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List

from ..image_canvas import resolve_image_path
from ..model_backends import ApiBackend, LocalBackend, ModelBackend
from .generator import generate_single_blueprint
from .io_utils import dump_jsonl, load_jsonl
from .review import review_blueprint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate task blueprints for Canvas multi-turn SFT pipeline.")
    parser.add_argument("--input-jsonl", required=True, help="Seed reasoning data in JSONL format.")
    parser.add_argument("--output-jsonl", required=True, help="Accepted blueprint output JSONL.")
    parser.add_argument("--rejected-jsonl", default="", help="Rejected records output JSONL.")
    parser.add_argument("--backend", choices=["api", "local"], required=True)
    parser.add_argument("--model", required=True, help="Model name (api) or model path (local).")
    parser.add_argument("--base-url", default="", help="OpenAI-compatible base URL for API backend.")
    parser.add_argument("--api-key", default="", help="API key for API backend. Defaults to OPENAI_API_KEY env.")
    parser.add_argument("--max-retries", type=int, default=2, help="Retries when validation fails.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start", type=int, default=0, help="Start index (inclusive) for slicing input data.")
    parser.add_argument("--end", type=int, default=-1, help="End index (exclusive). Use -1 for dataset end.")
    parser.add_argument("--limit", type=int, default=-1, help="Only process first N samples.")
    parser.add_argument("--enable-review", action="store_true", help="Enable committee review stage.")
    parser.add_argument("--committee-size", type=int, default=3)
    parser.add_argument("--min-total-score", type=int, default=3, help="Min review total score (0-4) for a vote.")
    parser.add_argument(
        "--enable-exec-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable APIGen-style execution validation in stage-1.",
    )
    parser.add_argument(
        "--require-gt-action-plan",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require optional_gt_action_plan when execution check is enabled.",
    )
    parser.add_argument("--max-plan-steps", type=int, default=12, help="Max steps for optional_gt_action_plan.")
    parser.add_argument("--local-max-new-tokens", type=int, default=1024)
    parser.add_argument("--local-device", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def build_backend(args: argparse.Namespace) -> ModelBackend:
    if args.backend == "api":
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("Missing API key. Provide --api-key or OPENAI_API_KEY.")
        return ApiBackend(model=args.model, api_key=api_key, base_url=args.base_url or None)

    return LocalBackend(
        model_path=args.model,
        device=args.local_device,
        max_new_tokens=args.local_max_new_tokens,
        trust_remote_code=args.trust_remote_code,
    )


def run(args: argparse.Namespace) -> int:
    random.seed(args.seed)

    in_path = Path(args.input_jsonl)
    out_path = Path(args.output_jsonl)
    rej_path = Path(args.rejected_jsonl) if args.rejected_jsonl else None

    all_seeds = load_jsonl(in_path)
    total_input = len(all_seeds)

    if args.start < 0:
        raise ValueError("--start must be >= 0")
    if args.end != -1 and args.end < args.start:
        raise ValueError("--end must be -1 or >= --start")

    end = total_input if args.end == -1 else min(args.end, total_input)
    start = min(args.start, total_input)

    indexed_seeds = list(enumerate(all_seeds))[start:end]
    if args.limit > 0:
        indexed_seeds = indexed_seeds[: args.limit]

    backend = build_backend(args)

    accepted_rows: List[Dict[str, Any]] = []
    rejected_rows: List[Dict[str, Any]] = []
    used_task_ids = set()

    for idx, (source_idx, seed) in enumerate(indexed_seeds):
        seed_input = dict(seed)
        raw_image_path = str(seed_input.get("image_path", "") or "").strip()
        if raw_image_path:
            seed_input["image_path"] = resolve_image_path(raw_image_path, base_dir=in_path.parent)

        fallback_task_id = str(seed_input.get("task_id") or f"task_{source_idx:06d}")
        bp, errors, raw, exec_summary = generate_single_blueprint(
            backend=backend,
            seed_item=seed_input,
            fallback_task_id=fallback_task_id,
            max_retries=args.max_retries,
            temperature=args.temperature,
            enable_exec_check=args.enable_exec_check,
            require_gt_action_plan=args.require_gt_action_plan,
            max_plan_steps=args.max_plan_steps,
        )

        if bp is None:
            rejected_rows.append(
                {
                    "task_id": fallback_task_id,
                    "seed": seed_input,
                    "status": "generation_failed",
                    "errors": errors,
                    "raw": raw,
                    "stage1_checks": {"exec_check": exec_summary},
                }
            )
            print(f"[REJECT][{fallback_task_id}] generation failed: {errors}")
            continue

        # Canonicalize task_id to the source seed id instead of trusting model-generated ids.
        bp["task_id"] = fallback_task_id
        final_task_id = fallback_task_id
        if final_task_id in used_task_ids:
            final_task_id = f"{final_task_id}__idx{source_idx:06d}"
            bp["task_id"] = final_task_id
        used_task_ids.add(final_task_id)

        review_result: Dict[str, Any] = {"enabled": False}
        if args.enable_review:
            ok, reviews = review_blueprint(
                backend=backend,
                blueprint=bp,
                committee_size=args.committee_size,
                min_total_score=args.min_total_score,
                temperature=args.temperature,
            )
            review_result = {"enabled": True, "pass": ok, "reviews": reviews}
            if not ok:
                rejected_rows.append(
                    {
                        "task_id": final_task_id,
                        "seed": seed_input,
                        "status": "review_rejected",
                        "errors": ["committee rejected"],
                        "blueprint": bp,
                        "review": review_result,
                        "stage1_checks": {"exec_check": exec_summary},
                    }
                )
                print(f"[REJECT][{fallback_task_id}] committee rejected")
                continue

        accepted_row: Dict[str, Any] = {
            "task_id": final_task_id,
            "blueprint": bp,
            "seed_meta": {"source_index": source_idx, "slice_index": idx},
            "review": review_result,
            "stage1_checks": {"exec_check": exec_summary},
        }
        if "reference_answer" in seed:
            accepted_row["reference_answer"] = seed.get("reference_answer", "")
        if "answer" in seed:
            accepted_row["answer"] = seed.get("answer", "")
        if "category" in seed:
            accepted_row["category"] = seed.get("category", "")
        if "source_meta" in seed:
            accepted_row["source_meta"] = seed.get("source_meta", {})

        accepted_rows.append(accepted_row)
        print(f"[ACCEPT][{fallback_task_id}]")

    dump_jsonl(out_path, accepted_rows)
    if rej_path is not None:
        dump_jsonl(rej_path, rejected_rows)

    print(
        json.dumps(
            {
                "input_total": total_input,
                "input_selected": len(indexed_seeds),
                "slice_start": start,
                "slice_end": end,
                "accepted": len(accepted_rows),
                "rejected": len(rejected_rows),
                "output_jsonl": str(out_path),
                "rejected_jsonl": str(rej_path) if rej_path else "",
            },
            ensure_ascii=False,
        )
    )
    return 0


def main() -> int:
    return run(parse_args())
