from __future__ import annotations

import argparse

from .collector import run_rollout_collection


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-2 rollout collector for Canvas multi-turn SFT.")
    parser.add_argument("--input-jsonl", required=True, help="Blueprint JSONL from stage-1.")
    parser.add_argument("--output-jsonl", required=True, help="Collected rollout output JSONL.")
    parser.add_argument("--rejected-jsonl", default="", help="Rejected rollout output JSONL.")

    parser.add_argument("--backend", choices=["api", "local"], required=True)
    parser.add_argument("--model", required=True, help="Model name (api) or model path (local).")
    parser.add_argument("--base-url", default="", help="OpenAI-compatible API base URL.")
    parser.add_argument("--api-key", default="", help="API key for api backend.")
    parser.add_argument("--api-timeout", type=int, default=120)

    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=1024, help="Per assistant turn max tokens.")
    parser.add_argument("--max-tools-per-turn", type=int, default=8)
    parser.add_argument("--force-final-retries", type=int, default=2)
    parser.add_argument(
        "--plan-hint-retries",
        type=int,
        default=2,
        help="Max retries to inject validated optional_gt_action_plan hints when model outputs no tool call.",
    )
    parser.add_argument(
        "--enable-answer-judge",
        dest="enable_answer_judge",
        action="store_true",
        default=True,
        help="Use LLM semantic equivalence judge for boxed content when strict match fails.",
    )
    parser.add_argument(
        "--no-enable-answer-judge",
        dest="enable_answer_judge",
        action="store_false",
        help="Disable LLM semantic equivalence judge.",
    )
    parser.add_argument(
        "--answer-judge-temperature",
        type=float,
        default=0.0,
        help="Answer judge decoding temperature.",
    )
    parser.add_argument(
        "--answer-judge-max-tokens",
        type=int,
        default=256,
        help="Answer judge max response tokens.",
    )

    parser.add_argument("--critic-backend", choices=["api", "local"], default="", help="Critic backend. Defaults to --backend.")
    parser.add_argument("--critic-model", default="", help="Critic model. Defaults to --model.")
    parser.add_argument("--critic-base-url", default="", help="Critic API base URL. Defaults to --base-url.")
    parser.add_argument("--critic-api-key", default="", help="Critic API key. Defaults to --api-key.")
    parser.add_argument("--critic-api-timeout", type=int, default=-1, help="Critic timeout. Defaults to --api-timeout.")
    parser.add_argument("--critic-temperature", type=float, default=0.0)
    parser.add_argument("--critic-max-tokens", type=int, default=512)
    parser.add_argument(
        "--critic-local-max-new-tokens",
        type=int,
        default=-1,
        help="Critic local max new tokens. Defaults to --local-max-new-tokens.",
    )
    parser.add_argument("--critic-local-device", default="", help="Critic local device. Defaults to --local-device.")
    parser.add_argument("--critic-trust-remote-code", action="store_true")

    parser.add_argument(
        "--render-dir",
        default="",
        help="Rendered tool image directory. Default: /tmp/canvas_rollout_renders",
    )
    parser.add_argument(
        "--success-only",
        action="store_true",
        help="Deprecated: hard filter is always enabled and only passed rollouts are kept.",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=-1, help="Only process first N lines.")

    parser.add_argument("--local-max-new-tokens", type=int, default=1024)
    parser.add_argument("--local-device", default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")

    return parser.parse_args()


def main() -> int:
    return run_rollout_collection(parse_args())
