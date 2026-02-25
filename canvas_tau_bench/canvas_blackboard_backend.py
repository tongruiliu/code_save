from __future__ import annotations

import os
import warnings
from typing import Optional

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from playwright.sync_api import sync_playwright


# Keep parsing SVG-in-HTML noise out of logs.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)


_DEFAULT_STATE = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Notebook State</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; }
    body {
      font-family: sans-serif;
      line-height: 1.6;
      background-color: #f4f7f9;
      color: #333;
      margin: 0;
      padding: 20px;
    }
    main {
      max-width: 800px;
      margin: 20px auto;
      padding: 20px;
      background-color: #fff;
      border-radius: 8px;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    img { max-width: 100%; height: auto; display: block; }
  </style>
</head>
<body>
  <div id="root"></div>
</body>
</html>
"""


class Blackboard:
    """Canvas-CoT style notebook backend for HTML/SVG CRUD and rendering."""

    def __init__(self, initial_svg: Optional[str] = None):
        try:
            BeautifulSoup("<b></b>", "lxml")
            self.parser = "lxml"
        except Exception:
            self.parser = "html.parser"

        self.state = _DEFAULT_STATE
        if initial_svg:
            self.update_state(action="insert_element", attrs={"fragment": initial_svg, "rootId": None})

    def _allow_external_images(self) -> bool:
        return os.environ.get("BLACKBOARD_ALLOW_EXTERNAL_IMAGES", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
            "on",
        }

    def _is_external_url(self, url: str) -> bool:
        u = (url or "").strip().lower()
        return u.startswith("http://") or u.startswith("https://")

    def _sanitize_external_images(self, fragment_soup: BeautifulSoup) -> None:
        """Replace external image URLs to keep rendering deterministic/offline-safe."""
        if self._allow_external_images():
            return

        for img in list(fragment_soup.find_all("img")):
            src = img.get("src")
            if isinstance(src, str) and self._is_external_url(src):
                replacement = fragment_soup.new_tag(
                    "div",
                    attrs={
                        "style": "color:#ED2633;font-size:14px;padding:8px 0;word-break:break-word;"
                    },
                )
                replacement.string = f"[blocked external image] {src}"
                img.replace_with(replacement)

        for svg_img in list(fragment_soup.find_all("image")):
            href = svg_img.get("href") or svg_img.get("xlink:href")
            if isinstance(href, str) and self._is_external_url(href):
                try:
                    x = svg_img.get("x") or "0"
                    y = svg_img.get("y") or "0"
                    text = fragment_soup.new_tag("text")
                    text["x"] = str(x)
                    text["y"] = str(y)
                    text["fill"] = "#ED2633"
                    text.string = "[blocked external svg image]"
                    svg_img.replace_with(text)
                except Exception:
                    svg_img.decompose()

    def update_state(self, action: str, attrs: dict):
        """Apply one CRUD action to notebook state."""
        soup = BeautifulSoup(self.state, self.parser)

        if action == "insert_element":
            fragment_str = attrs.get("fragment")
            if not fragment_str:
                return

            fragment_soup = BeautifulSoup(fragment_str, self.parser)
            self._sanitize_external_images(fragment_soup)
            new_element = fragment_soup.find()
            if not new_element:
                return

            if not new_element.get("id"):
                new_element["id"] = "initial_svg" if new_element.name == "svg" else f"elem_{len(soup.find_all())}"

            root_id = attrs.get("rootId")
            parent_element = soup.find(id=root_id) if root_id else soup.body
            if not parent_element:
                if root_id == "root":
                    new_root = soup.new_tag("div", id="root")
                    soup.body.append(new_root)
                    parent_element = new_root
                else:
                    parent_element = soup.body

            before_id = attrs.get("beforeId")
            if before_id:
                before_element = soup.find(id=before_id)
                if before_element:
                    before_element.insert_before(new_element)
                else:
                    parent_element.append(new_element)
            else:
                parent_element.append(new_element)

        elif action == "modify_element":
            target_id = attrs.get("targetId")
            attributes_to_update = attrs.get("attrs")
            if not target_id or not attributes_to_update:
                return

            target_element = soup.find(id=target_id)
            if target_element:
                for key, value in attributes_to_update.items():
                    if key == "text":
                        target_element.string = str(value)
                    else:
                        target_element[key] = str(value)

        elif action == "remove_element":
            target_id = attrs.get("targetId")
            if not target_id:
                return
            target_element = soup.find(id=target_id)
            if target_element:
                target_element.decompose()

        elif action in {"clear_element", "clear"}:
            if soup.body:
                soup.body.clear()
                new_root = soup.new_tag("div", id="root")
                soup.body.append(new_root)
            else:
                self.__init__()
                return

        elif action == "replace_element":
            target_id = attrs.get("targetId")
            fragment_str = attrs.get("fragment")
            if not target_id or not fragment_str:
                return

            target_element = soup.find(id=target_id)
            if target_element:
                new_element = BeautifulSoup(fragment_str, self.parser).find()
                if new_element:
                    target_element.replace_with(new_element)

        else:
            return

        self.state = str(soup)

    def render_state(self, output_path: str = "output.png") -> str:
        """Render current state to image using Playwright."""
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        try:
            soup = BeautifulSoup(self.state, self.parser)
            if soup.body:
                children = [child for child in soup.body.contents if getattr(child, "name", None)]
                if len(children) > 3:
                    for child in children[:-3]:
                        child.decompose()

            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page(
                    viewport={"width": 500, "height": 500},
                    device_scale_factor=2,
                )
                page.set_content(str(soup))
                page.screenshot(path=output_path, full_page=True)
                browser.close()
                return "tool execute success"
        except Exception as exc:
            return f"tool execute failed: {exc}"

