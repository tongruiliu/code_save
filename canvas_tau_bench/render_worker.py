from __future__ import annotations

import argparse
import os
import sys

from playwright.sync_api import sync_playwright


def render_html_to_png(html_text: str, output_path: str) -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page(
                viewport={"width": 1200, "height": 900},
                device_scale_factor=2,
            )
            try:
                page.set_content(html_text, wait_until="load")
                page.screenshot(path=output_path, full_page=True)
            finally:
                page.close()
        finally:
            browser.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render HTML to PNG using Playwright in an isolated process.")
    parser.add_argument("--html-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    with open(args.html_path, "r", encoding="utf-8") as f:
        html_text = f.read()

    out_dir = os.path.dirname(args.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    render_html_to_png(html_text=html_text, output_path=args.output_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise
