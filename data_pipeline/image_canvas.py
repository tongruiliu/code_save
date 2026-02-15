from __future__ import annotations

import base64
import html
import mimetypes
import re
from pathlib import Path
from typing import Optional


_MAIN_SVG_OPEN_RE = re.compile(r"<svg\b[^>]*\bid\s*=\s*['\"]main_svg['\"][^>]*>", flags=re.IGNORECASE)
_BASE_IMAGE_ID_RE = re.compile(r"\bid\s*=\s*['\"]base_image['\"]", flags=re.IGNORECASE)


def resolve_image_path(image_path: str, base_dir: Optional[Path] = None) -> str:
    raw = (image_path or "").strip()
    if not raw:
        return ""
    p = Path(raw)
    if p.is_absolute():
        return str(p)
    if base_dir is not None:
        return str((base_dir / p).resolve())
    return str(p.resolve())


def path_to_image_uri(image_path: str) -> str:
    raw = (image_path or "").strip()
    if not raw:
        return ""
    lower = raw.lower()
    if lower.startswith("data:") or lower.startswith("file://") or lower.startswith("http://") or lower.startswith("https://"):
        return raw

    p = Path(raw).resolve()
    if p.exists() and p.is_file():
        mime, _ = mimetypes.guess_type(str(p))
        if not mime:
            mime = "image/png"
        payload = base64.b64encode(p.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{payload}"

    return p.as_uri()


def build_base_init_svg(image_path: str, width: int = 800, height: int = 600) -> str:
    image_uri = path_to_image_uri(image_path)
    if image_uri:
        href = html.escape(image_uri, quote=True)
        return (
            f'<svg id="main_svg" width="{int(width)}" height="{int(height)}" '
            f'xmlns="http://www.w3.org/2000/svg">'
            f'<image id="base_image" href="{href}" x="0" y="0" width="{int(width)}" height="{int(height)}" '
            'preserveAspectRatio="xMidYMid meet"/></svg>'
        )
    return (
        f'<svg id="main_svg" width="{int(width)}" height="{int(height)}" '
        'xmlns="http://www.w3.org/2000/svg"></svg>'
    )


def ensure_image_based_init_svg(init_svg: str, image_path: str, width: int = 800, height: int = 600) -> str:
    text = (init_svg or "").strip()
    if not text:
        return build_base_init_svg(image_path=image_path, width=width, height=height)

    image_uri = path_to_image_uri(image_path)
    if not image_uri:
        return text

    if _BASE_IMAGE_ID_RE.search(text):
        return text

    m = _MAIN_SVG_OPEN_RE.search(text)
    if not m:
        return build_base_init_svg(image_path=image_path, width=width, height=height)

    href = html.escape(image_uri, quote=True)
    image_tag = (
        f'<image id="base_image" href="{href}" x="0" y="0" width="{int(width)}" height="{int(height)}" '
        'preserveAspectRatio="xMidYMid meet"/>'
    )
    return text[: m.end()] + image_tag + text[m.end() :]
