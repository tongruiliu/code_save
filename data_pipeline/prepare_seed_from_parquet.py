#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_INIT_SVG = '<svg id="main_svg" width="800" height="600" xmlns="http://www.w3.org/2000/svg"></svg>'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert parquet dataset to stage-1 seed JSONL.")
    parser.add_argument("--input-parquet", required=True, help="Input parquet path.")
    parser.add_argument("--output-jsonl", required=True, help="Output seed JSONL path.")
    parser.add_argument(
        "--image-dir",
        default="",
        help="Directory to save decoded images. Default: <output-jsonl-dir>/images",
    )
    parser.add_argument("--task-prefix", default="parquet_task", help="Task id prefix.")
    parser.add_argument("--limit", type=int, default=-1, help="Only process first N rows.")
    parser.add_argument(
        "--keep-missing-image",
        action="store_true",
        help="Keep rows without image bytes (image_path will be empty).",
    )
    parser.add_argument(
        "--absolute-image-path",
        action="store_true",
        help="Write absolute image_path in JSONL. Default writes path relative to output JSONL dir.",
    )
    parser.add_argument(
        "--default-init-svg",
        default=DEFAULT_INIT_SVG,
        help="Default init_svg for every seed row.",
    )
    return parser.parse_args()


def _as_dict_image(image_value: Any) -> Dict[str, Any]:
    if image_value is None:
        return {}
    if isinstance(image_value, dict):
        return image_value
    return {}


def _safe_filename(row_idx: int, raw_name: str) -> str:
    base = Path(raw_name or "").name
    if not base:
        base = f"{row_idx:06d}.png"
    return f"{row_idx:06d}_{base}"


def _ensure_pyarrow() -> Any:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except Exception as exc:
        raise RuntimeError("pyarrow is required. Please run in an environment with pyarrow installed.") from exc
    return pq


def _to_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return str(x)


def _format_answer_field(raw_answer: str) -> str:
    text = _to_str(raw_answer).strip()
    if not text:
        return ""

    m = re.search(r"<answer>(.*?)</answer>", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        inner = m.group(1).strip()
        if "\\boxed{" in inner:
            return f"<answer>{inner}</answer>"
        return f"<answer>\\boxed{{{inner}}}</answer>"

    if "\\boxed{" in text:
        return f"<answer>{text}</answer>"
    return f"<answer>\\boxed{{{text}}}</answer>"


def run(args: argparse.Namespace) -> int:
    pq = _ensure_pyarrow()

    input_path = Path(args.input_parquet)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_dir = Path(args.image_dir) if args.image_dir else (output_path.parent / "images")
    image_dir.mkdir(parents=True, exist_ok=True)

    table = pq.read_table(str(input_path))
    rows: List[Dict[str, Any]] = table.to_pylist()
    if args.limit > 0:
        rows = rows[: args.limit]

    kept = 0
    skipped_missing_image = 0
    output_rows: List[Dict[str, Any]] = []

    for idx, row in enumerate(rows):
        question = _to_str(row.get("question")).strip()
        answer = _to_str(row.get("answer")).strip()
        category = _to_str(row.get("catagory")).strip()
        answer_field = _format_answer_field(answer)

        image_obj = _as_dict_image(row.get("image"))
        image_bytes = image_obj.get("bytes")
        image_raw_name = _to_str(image_obj.get("path")).strip()

        image_path_value = ""
        if isinstance(image_bytes, (bytes, bytearray)) and len(image_bytes) > 0:
            filename = _safe_filename(idx, image_raw_name)
            image_path = image_dir / filename
            image_path.write_bytes(bytes(image_bytes))
            if args.absolute_image_path:
                image_path_value = str(image_path.resolve())
            else:
                image_path_value = str(image_path.relative_to(output_path.parent))
        else:
            if not args.keep_missing_image:
                skipped_missing_image += 1
                continue

        seed_row = {
            "task_id": f"{args.task_prefix}_{idx:06d}",
            "question": question,
            "image_path": image_path_value,
            "init_svg": args.default_init_svg,
            "answer": answer_field,
            "category": category,
            "reference_answer": answer,
            "source_meta": {
                "source_file": str(input_path),
                "row_index": idx,
                "raw_image_name": image_raw_name,
            },
        }
        output_rows.append(seed_row)
        kept += 1

    with output_path.open("w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "input_rows": len(rows),
                "kept_rows": kept,
                "skipped_missing_image": skipped_missing_image,
                "output_jsonl": str(output_path),
                "image_dir": str(image_dir),
            },
            ensure_ascii=False,
        )
    )
    return 0


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
